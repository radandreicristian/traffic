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
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.nn.conv import GATConv
from src.model.gman.util.knn_rewire import knn_rewire


class GATWrapper(nn.Module):
    def __init__(
        self, d_hidden_feat, d_hidden_pos, n_heads, edge_index, edge_attr, k=None
    ) -> None:
        super(GATWrapper, self).__init__()
        self.d_hidden = d_hidden_feat + d_hidden_pos

        assert self.d_hidden % n_heads == 0, (
            "Hidden size not divisible by number of " "heads."
        )

        d_head = self.d_hidden // n_heads

        self.n_heads = n_heads
        self.fc_in = nn.Linear(in_features=self.d_hidden, out_features=self.d_hidden)
        self.gat_layer = GATConv(
            in_channels=self.d_hidden,
            out_channels=d_head,
            heads=n_heads,
            dropout=0.5,
            edge_dim=1,
        )
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.fc_out = nn.Linear(in_features=self.d_hidden, out_features=d_hidden_feat)

    def forward(self, x: torch.Tensor, ste):
        b, l, n, d = x.shape

        # features (batch, seq, n_nodes, d_hidden_feat+d_hidden_pos)
        h = torch.cat([x, ste], dim=-1)

        h = rearrange(h, "b l n d -> (b l) n d")

        h = torch.stack(
            [
                self.gat_layer(
                    g,
                    edge_index=self.edge_index,
                    edge_attr=self.edge_attr,
                )
                for g in h
            ]
        )

        h = rearrange(h, "(b l) n d -> b l n d", b=b)
        return self.fc_out(h)


class SpatioTemporalGraphAttention(nn.Module):
    def __init__(
        self, n_heads, d_hidden, d_hidden_pos, bn_decay, edge_index, edge_attr
    ):
        super(SpatioTemporalGraphAttention, self).__init__()

        self.spatial_attention = GATWrapper(
            d_hidden_feat=d_hidden,
            d_hidden_pos=d_hidden_pos,
            n_heads=n_heads,
            edge_index=edge_index,
            edge_attr=edge_attr,
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


class GATMAN(nn.Module):
    def __init__(
        self, opt: dict, dataset: torch_geometric.data.Dataset, device: torch.device
    ):
        super(GATMAN, self).__init__()

        self.device = device
        self.d_hidden = opt.get("d_hidden")
        self.d_hidden_pos = opt.get("d_hidden_pos")
        self.n_heads = opt.get("n_heads")
        self.bn_decay = opt.get("bn_decay", 0.9)

        self.n_previous = opt.get("n_previous_steps")
        self.n_future = opt.get("n_future_steps")
        self.n_blocks = opt.get("n_blocks")

        self.edge_index, self.edge_attr = map(
            lambda x: x.to(self.device), dataset.get_adjacency_matrix()
        )

        self.rewire = opt.get("knn_rewire")
        if self.rewire:
            k = opt.get("knn_rewire_k")
            self.edge_index, self.edge_attr = map(
                lambda x: x.to(self.device),
                knn_rewire(self.edge_index, k, self.edge_attr),
            )

        # Todo - This will require grad later)
        self.positional_embeddings = nn.Parameter(
            data=opt.get("positional_embeddings"), requires_grad=False
        )
        self.st_embedding = SpatioTemporalEmbedding(
            d_hidden=self.d_hidden_pos, bn_decay=self.bn_decay
        )

        self.encoder = nn.ModuleList(
            [
                SpatioTemporalGraphAttention(
                    n_heads=self.n_heads,
                    d_hidden=self.d_hidden,
                    d_hidden_pos=self.d_hidden_pos,
                    bn_decay=self.bn_decay,
                    edge_index=self.edge_index,
                    edge_attr=self.edge_attr,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.transform_attention = TransformAttention(
            d_hidden=self.d_hidden, n_heads=self.n_heads, bn_decay=self.bn_decay
        )
        self.decoder = nn.ModuleList(
            [
                SpatioTemporalGraphAttention(
                    n_heads=self.n_heads,
                    d_hidden=self.d_hidden,
                    d_hidden_pos=self.d_hidden_pos,
                    bn_decay=self.bn_decay,
                    edge_index=self.edge_index,
                    edge_attr=self.edge_attr,
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
        self, x_signal: torch.Tensor, x_temporal: torch.Tensor, y_temporal: torch.Tensor
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
