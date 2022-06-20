import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_geometric
from einops import rearrange
from reformer_pytorch import LSHSelfAttention
from torch import Tensor

from src.model.gman.gman_blocks import (
    FullyConnected,
    TemporalAttention,
    GatedFusion,
    SpatioTemporalEmbedding,
    TransformAttention,
)


class LshSelfAttention(nn.Module):
    def __init__(self,
                 d_hidden: int,
                 n_heads: int,
                 p_dropout: float,
                 use_mask: bool = False,
                 bucket_size: int = 16,
                 n_hashes: int = 4
                 ):
        super(LshSelfAttention, self).__init__()
        assert d_hidden % n_heads == 0, "Hidden dimension must be divisible by n_heads."

        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.lsh_attention = LSHSelfAttention(dim=d_hidden,
                                              heads=n_heads,
                                              bucket_size=bucket_size,
                                              n_hashes=n_hashes,
                                              causal=False)

    def forward(self, x: torch.Tensor):
        # x (b, l, d)
        b, l, d = x.shape

        # Seq len must be divisble with 2*bucket_size. Pad to match lens
        padding = 2 * self.bucket_size - l % (2 * self.bucket_size)

        if padding != 0:
            pad_tensor = torch.zeros((b, padding, d)).to(x.device)
            x = torch.cat([x, pad_tensor], dim=-2)
        x = self.lsh_attention(x)
        return x[:, :-padding, :]


class LshSpatialAttention(nn.Module):
    def __init__(
            self,
            d_hidden_feat,
            d_hidden_pos,
            n_heads,
            n_nodes,
            p_dropout,
    ) -> None:
        super(LshSpatialAttention, self).__init__()

        assert d_hidden_feat % n_heads == 0, (
            "Hidden size not divisible by number of " "heads."
        )

        self.n_heads = n_heads
        self.n_nodes = n_nodes
        d_hidden = d_hidden_feat + d_hidden_pos
        self.projection = nn.Linear(in_features=d_hidden, out_features=d_hidden_feat)

        self.linear_self_attention = LshSelfAttention(
            d_hidden=d_hidden_feat, n_heads=self.n_heads, p_dropout=p_dropout
        )

    def forward(self, x: torch.Tensor, ste):
        b, l, n, d = x.shape
        # features (batch, seq, n_nodes, d_hidden_feat+d_hidden_pos)
        h = torch.cat([x, ste], dim=-1)
        h = rearrange(h, "b l n d -> (b l) n d")
        h = self.projection(h)
        h = self.linear_self_attention(h)
        h = rearrange(h, "(b l) n d -> b l n d", b=b)
        return h


class FastSpatioTemporalBlock(nn.Module):
    def __init__(
        self,
        n_heads,
        d_hidden,
        d_hidden_pos,
        bn_decay,
        n_nodes,
        p_dropout,
    ):
        super(FastSpatioTemporalBlock, self).__init__()

        self.spatial_attention = LshSpatialAttention(
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


class LshGMAN(nn.Module):
    def __init__(
        self, opt: dict, dataset: torch_geometric.data.Dataset, device: torch.device
    ):
        super(LshGMAN, self).__init__()

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
                FastSpatioTemporalBlock(
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
                FastSpatioTemporalBlock(
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
