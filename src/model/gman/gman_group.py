import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_geometric
from einops import rearrange
from torch import Tensor

from src.model.gman.gman_blocks import (
    FullyConnected,
    TemporalAttention,
    GatedFusion,
    SpatioTemporalEmbedding,
    TransformAttention,
)


class GroupAttention(nn.Module):
    def __init__(self,
                 d_hidden,
                 d_hidden_features):
        super(GroupAttention, self).__init__()

        self.q_intra = nn.Linear(in_features=d_hidden, out_features=d_hidden_features)
        self.k_intra = nn.Linear(in_features=d_hidden, out_features=d_hidden_features)
        self.v_intra = nn.Linear(in_features=d_hidden, out_features=d_hidden_features)

        self.q_inter = nn.Linear(in_features=d_hidden_features,
                                 out_features=d_hidden_features)
        self.k_inter = nn.Linear(in_features=d_hidden_features,
                                 out_features=d_hidden_features)
        self.v_inter = nn.Linear(in_features=d_hidden_features,
                                 out_features=d_hidden_features)

        self.scale = self.d_head ** 0.5
        self.to_out_intra = nn.Linear(in_features=d_hidden_features,
                                      out_features=d_hidden_features)

        self.to_out_inter = nn.Linear(in_features=d_hidden_features,
                                      out_features=d_hidden_features)

        self.group_pool = nn.MaxPool1d(kernel_size=d_hidden_features)
        self.m = 37  # This is from the paper :shrug:

    def forward_single_intra(self, x):
        q = self.q_intra(x)
        k = self.k_intra(x)
        v = self.v_intra(x)
        q, k, v = (rearrange(i,
                             "b n (h d) -> b h n d",
                             h=self.n_heads) for i in (q, k, v))

        attention = f.softmax((q @ k.transpose(-1, -2) * self.scale), dim=-1)
        v = attention @ v
        return self.to_out_intra(rearrange(v, "b h n d -> b n (h d)"))

    def forward_single_inter(self, x):
        q = self.q_inter(x)
        k = self.k_inter(x)
        v = self.v_inter(x)
        q, k, v = (rearrange(i,
                             "b n (h d) -> b h n d",
                             h=self.n_heads) for i in (q, k, v))

        attention = f.softmax((q @ k.transpose(-1, -2) * self.scale), dim=-1)
        v = attention @ v
        return self.to_out_inter(rearrange(v, "b h n d -> b n (h d)"))

    def forward(self, x, **kwargs):
        b, _, d = x.shape
        partitions = kwargs.get("partitions")
        chunk_size = len(partitions[0])
        odd_chunk_size = len(partitions[-1])
        intra_group_attentions = []

        # Intra-group Attention
        for partition in partitions:

            # x_ [B, N/K, D]
            x_ = x[:, partition, :]
            group_size = len(partition)

            # Pad the smaller chunk if there is one
            if group_size == odd_chunk_size:
                padding = chunk_size - odd_chunk_size
                pad_tensor = torch.zeros((b, padding, d)).to(x.device)
                x_ = torch.cat([x_, pad_tensor], dim=1)

            # (B, N/K, D)
            group_attention = self.forward_single_intra(x_)
            intra_group_attentions.append(group_attention)

        # attentive_group (B, N/K, d)
        inter_group_features = []
        for attentive_group in intra_group_attentions:
            attentive_group = attentive_group.transpose(-1, -2)
            # pooled_group_features (B, d)
            pooled_group_features = self.group_pool(attentive_group).squeeze()
            inter_group_features.append(pooled_group_features)

        # inter_group_features (B, K, d)
        inter_group_features = torch.stack(inter_group_features, dim=-2)
        inter_group_attentive_features = self.forward_single_inter(inter_group_features)

        # group (B, N/K, d)
        for index, group in enumerate(intra_group_attentions):
            group += inter_group_attentive_features[:, index, :]

        features = torch.cat(intra_group_attentions, dim=-2)

        return features[:, torch.argsort(partitions), :]


class GroupSpatialAttention(nn.Module):
    def __init__(
            self,
            d_hidden_feat,
            d_hidden_pos,
            n_heads,
            n_nodes,
            p_dropout,
    ) -> None:
        super(GroupSpatialAttention, self).__init__()
        self.d_hidden = d_hidden_feat + d_hidden_pos

        assert self.d_hidden % n_heads == 0, (
            "Hidden size not divisible by number of " "heads."
        )

        self.n_heads = n_heads
        self.n_nodes = n_nodes

        self.fc_out = nn.Linear(in_features=self.d_hidden, out_features=d_hidden_feat)

    def forward(self, x: torch.Tensor, ste):
        b, l, n, d = x.shape
        # features (batch, seq, n_nodes, d_hidden_feat+d_hidden_pos)
        h = torch.cat([x, ste], dim=-1)
        h = rearrange(h, "b l n d -> (b l) n d")

        h = rearrange(h, "(b l) n d -> b l n d", b=b)
        return self.fc_out(h)


class GroupSpatioTemporalBlock(nn.Module):
    def __init__(
            self,
            n_heads,
            d_hidden,
            d_hidden_pos,
            bn_decay,
            n_nodes,
            p_dropout,
    ):
        super(GroupSpatioTemporalBlock, self).__init__()

        self.spatial_attention = GroupSpatialAttention(
            d_hidden_feat=d_hidden,
            d_hidden_pos=d_hidden_pos,
            n_heads=n_heads,
            n_nodes=n_nodes,
            p_dropout=p_dropout,
        )
        self.temporal_attention = TemporalAttention(
            d_hidden=d_hidden,
            n_heads=n_heads,
            bn_decay=bn_decay,
            d_hidden_pos=d_hidden_pos,
        )
        self.gated_fusion = GatedFusion(d_hidden=d_hidden, bn_decay=bn_decay)

    def forward(self, x, ste):
        h_spatial = self.spatial_attention(x, ste)
        h_temporal = self.temporal_attention(x, ste)
        h = self.gated_fusion(h_spatial, h_temporal)

        del h_spatial, h_temporal
        return x + h


class EfficientGMAN(nn.Module):
    def __init__(
            self, opt: dict, dataset: torch_geometric.data.Dataset, device: torch.device
    ):
        super(EfficientGMAN, self).__init__()

        self.device = device
        self.d_hidden = opt.get("d_hidden")
        self.d_hidden_pos = opt.get("d_hidden_pos")
        self.n_heads = opt.get("n_heads")
        self.bn_decay = opt.get("bn_decay", 0.9)

        self.n_previous = opt.get("n_previous_steps")
        self.n_future = opt.get("n_future_steps")
        self.n_blocks = opt.get("n_blocks")
        self.n_nodes = opt.get("n_nodes")

        self.positional_embeddings = nn.Parameter(
            data=opt.get("positional_embeddings"), requires_grad=False
        )
        self.st_embedding = SpatioTemporalEmbedding(
            d_hidden=self.d_hidden_pos, bn_decay=self.bn_decay
        )

        self.p_dropout = opt.get("p_dropout_model")
        self.encoder = nn.ModuleList(
            [
                LinearSpatioTemporalBlock(
                    n_heads=self.n_heads,
                    d_hidden=self.d_hidden,
                    d_hidden_pos=self.d_hidden_pos,
                    bn_decay=self.bn_decay,
                    n_nodes=self.n_nodes,
                    p_dropout=self.p_dropout,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.transform_attention = TransformAttention(
            d_hidden=self.d_hidden, n_heads=self.n_heads, bn_decay=self.bn_decay
        )
        self.decoder = nn.ModuleList(
            [
                LinearSpatioTemporalBlock(
                    n_heads=self.n_heads,
                    d_hidden=self.d_hidden,
                    d_hidden_pos=self.d_hidden_pos,
                    bn_decay=self.bn_decay,
                    n_nodes=self.n_nodes,
                    p_dropout=self.p_dropout,
                )
                for _ in range(self.n_blocks)
            ]
        )

        self.fc_in = FullyConnected(
            in_features=[1, self.d_hidden],
            out_features=[self.d_hidden, self.d_hidden],
            activations=[f.relu, None],
            bn_decay=self.bn_decay,
        )

        self.fc_out = FullyConnected(
            in_features=[self.d_hidden, self.d_hidden],
            out_features=[self.d_hidden, 1],
            activations=[f.relu, None],
            bn_decay=self.bn_decay,
        )

    def forward(
            self, x_signal: torch.Tensor, x_temporal: torch.Tensor,
            y_temporal: torch.Tensor
    ) -> Tensor:
        # x_signal: (batch_size, n_previous, n_nodes, 1)
        # x_temporal: (batch_size, n_previous, n_nodes, 2)
        # y_temporal: (batch_size, n_future, n_nodes, 2)

        x = self.fc_in(x_signal)

        # temporal_features (batch_size, n_previous+n_future, n_nodes, 31)
        temporal_features = torch.cat((x_temporal, y_temporal), dim=1)
        first_future_index = temporal_features.shape[1] // 2

        st_embeddings = self.st_embedding(
            spatial_embeddings=self.positional_embeddings,
            temporal_embeddings=temporal_features,
        )

        st_embeddings_previous = st_embeddings[:, :first_future_index, ...]
        st_embeddings_future = st_embeddings[:, first_future_index:, ...]

        for block in self.encoder:
            x = block(x, st_embeddings_previous)

        x = self.transform_attention(x, st_embeddings_previous, st_embeddings_future)

        for block in self.decoder:
            x = block(x, st_embeddings_future)

        x = torch.squeeze(self.fc_out(x), 3)
        return x

    def __repr__(self):
        return self.__class__.__name__
