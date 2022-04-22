"""Source:
https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/attention/gman.py"""
from typing import List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_geometric.data
from einops import rearrange
from torch import Tensor

from src.model import BaseGNN
from src.model.gman_blocks import (
    FullyConnected,
    SpatioTemporalEmbedding,
    TemporalAttention,
    GatedFusion,
    TransformAttention,
    SpatioTemporalAttention,
)
from src.model.ode.blocks import get_ode_block
from src.model.ode.blocks.funcs import get_ode_function
from src.util.utils import get_number_of_nodes


class SpatialDiffusionOde(BaseGNN):
    def __init__(
        self, opt: dict, n_nodes, edge_index, edge_attr, device: torch.device
    ) -> None:
        self.device = device
        self.edge_index, self.edge_attr = edge_index, edge_attr

        self.n_previous_steps = opt["n_previous_steps"]
        self.n_future_steps = opt["n_future_steps"]

        self.n_nodes = n_nodes

        super(SpatialDiffusionOde, self).__init__(opt=opt)

        self.function = get_ode_function(opt)

        time_tensor = torch.tensor([0, self.T])
        self.use_beltrami = opt.get("use_beltrami")

        d_hidden_pos = opt.get("d_hidden_pos")
        d_hidden_feat = opt.get("d_hidden")

        if self.use_beltrami:
            self.d_hidden = opt.get("d_hidden") + opt.get("d_hidden_pos")
            self.fc_pos = FullyConnected(
                in_features=d_hidden_pos,
                out_features=d_hidden_pos,
                activations=f.softplus,
                bn_decay=0.9,
            )
        self.fc_x = FullyConnected(
            in_features=d_hidden_feat,
            out_features=d_hidden_feat,
            activations=f.softplus,
            bn_decay=0.9,
        )

        self.ode_block = get_ode_block(opt)(
            ode_func=self.function,
            reg_funcs=self.reg_funcs,
            opt=opt,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            n_nodes=self.n_nodes,
            t=time_tensor,
            device=self.device,
        )

        self.cached_encoder_states = []
        self.reg_states: Optional[List[Any]] = None

    def forward(self, x, ste):
        batch_size, seq_len, n_nodes, d_hidden = x.shape

        if self.use_beltrami:
            # x (batch, seq, nodes, d_hidden_feat)
            x = self.fc_in(x)

            # pos (batch, seq, nodes, d_hidden_pos)
            pos = self.fc_pos(ste)

            # h (batch, seq, nodes, dh_feat+dh_pos=self.d_hidden)
            h = torch.cat([x, pos], dim=-1)
        else:
            h = self.fc_x(x)
        h = rearrange(
            h, "batch seq_len n_nodes d_hidden -> (batch seq_len) n_nodes d_hidden"
        )
        h = f.dropout(h, self.opt["p_dropout_model"], training=self.training)

        if self.opt["use_batch_norm"]:
            h = self.rearrange_batch_norm(h)

        if self.opt["use_augmentation"]:
            h = self.augment_up(h)

        self.ode_block.set_x0(h)

        if self.training and self.ode_block.n_reg > 0:
            z, self.reg_states = self.ode_block(h)
        else:
            z = self.ode_block(h)

        if self.opt["use_augmentation"]:
            z = self.augment_down(z)

        z = rearrange(
            z,
            "(batch seq_len) n_nodes d_hidden -> batch seq_len n_nodes d_hidden",
            batch=batch_size,
        )

        self.cached_encoder_states.append(z)

        return z
        # return self.fully_connected_out(z)


class SpatioTemporalAttentionOde(nn.Module):
    def __init__(
        self, n_heads, d_hidden, bn_decay, opt, edge_index, edge_attr, n_nodes, device
    ):
        super(SpatioTemporalAttentionOde, self).__init__()

        self.spatial_attention = SpatialDiffusionOde(
            opt=opt,
            edge_index=edge_index,
            edge_attr=edge_attr,
            n_nodes=n_nodes,
            device=device,
        )
        self.temporal_attention = TemporalAttention(
            d_hidden=d_hidden, n_heads=n_heads, bn_decay=bn_decay
        )
        self.gated_fusion = GatedFusion(d_hidden=d_hidden, bn_decay=bn_decay)

    def forward(self, x, ste):
        h_spatial = self.spatial_attention(x, ste)
        h_temporal = self.temporal_attention(x, ste)
        h = self.gated_fusion(h_spatial, h_temporal)

        del h_spatial, h_temporal
        return x + h


class GraphMultiAttentionNetOde(nn.Module):
    def __init__(
        self, opt: dict, dataset: torch_geometric.data.Dataset, device: torch.device
    ):
        super(GraphMultiAttentionNetOde, self).__init__()

        self.device = device
        self.d_hidden = opt.get("d_hidden")
        self.n_heads = opt.get("n_heads")
        self.bn_decay = opt.get("bn_decay", 0.9)

        self.n_previous = opt.get("n_previous_steps")
        self.n_future = opt.get("n_future_steps")
        self.n_blocks = opt.get("n_blocks")

        self.edge_index, self.edge_attr = dataset.get_adjacency_matrix()
        self.n_nodes = get_number_of_nodes(dataset=dataset, opt=opt)

        self.opt = opt
        # Todo - This will require grad later)
        self.positional_embeddings = nn.Parameter(
            data=opt.get("positional_embeddings"), requires_grad=False
        )
        self.st_embedding = SpatioTemporalEmbedding(
            d_hidden=self.d_hidden, bn_decay=self.bn_decay
        )

        self.encoder = nn.ModuleList(
            [
                SpatioTemporalAttentionOde(
                    n_heads=self.n_heads,
                    d_hidden=self.d_hidden,
                    bn_decay=self.bn_decay,
                    opt=self.opt,
                    edge_index=self.edge_index,
                    edge_attr=self.edge_attr,
                    n_nodes=self.n_nodes,
                    device=self.device,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.transform_attention = TransformAttention(
            d_hidden=self.d_hidden, n_heads=self.n_heads, bn_decay=self.bn_decay
        )
        self.decoder = nn.ModuleList(
            [
                SpatioTemporalAttentionOde(
                    n_heads=self.n_heads,
                    d_hidden=self.d_hidden,
                    bn_decay=self.bn_decay,
                    opt=self.opt,
                    edge_index=self.edge_index,
                    edge_attr=self.edge_attr,
                    n_nodes=self.n_nodes,
                    device=self.device,
                )
                for _ in range(self.n_blocks)
            ]
        )

        self.fc_in = FullyConnected(
            in_features=[1, self.d_hidden],
            out_features=[self.d_hidden, self.d_hidden],
            activations=[f.softplus, None],
            bn_decay=self.bn_decay,
        )

        self.fc_out = FullyConnected(
            in_features=[self.d_hidden, self.d_hidden],
            out_features=[self.d_hidden, 1],
            activations=[f.softplus, None],
            bn_decay=self.bn_decay,
        )

    def forward(
        self, x_signal: torch.Tensor, x_temporal: torch.Tensor, y_temporal: torch.Tensor
    ) -> Tensor:
        # x_signal: (batch_size, n_previous, n_nodes, 1)
        # x_temporal: (batch_size, n_previous, n_nodes, 2)
        # y_temporal: (batch_size, n_future, n_nodes, 2)

        # x (batch_size, n_previous, n_nodes, d_hidden)
        x = self.fc_in(x_signal)

        # temporal_features (batch_size, n_previous+n_future, n_nodes, 2)
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

        x = torch.squeeze(self.fc_out(x), dim=3)
        return x

    def __repr__(self):
        return self.__class__.__name__


class GraphMultiAttentionNetEncOde(nn.Module):
    def __init__(
        self, opt: dict, dataset: torch_geometric.data.Dataset, device: torch.device
    ):
        super(GraphMultiAttentionNetEncOde, self).__init__()

        self.device = device
        self.d_hidden = opt.get("d_hidden")
        self.n_heads = opt.get("n_heads")
        self.bn_decay = opt.get("bn_decay", 0.9)

        self.n_previous = opt.get("n_previous_steps")
        self.n_future = opt.get("n_future_steps")
        self.n_blocks_enc = opt.get("n_blocks_enc")
        self.n_blocks_dec = opt.get("n_blocks_dec")

        self.edge_index, self.edge_attr = dataset.get_adjacency_matrix()
        self.n_nodes = get_number_of_nodes(dataset=dataset, opt=opt)

        self.opt = opt
        # Todo - This will require grad later)
        self.positional_embeddings = nn.Parameter(
            data=opt.get("positional_embeddings"), requires_grad=False
        )
        self.st_embedding = SpatioTemporalEmbedding(
            d_hidden=self.d_hidden, bn_decay=self.bn_decay
        )

        self.encoder = nn.ModuleList(
            [
                SpatioTemporalAttentionOde(
                    n_heads=self.n_heads,
                    d_hidden=self.d_hidden,
                    bn_decay=self.bn_decay,
                    opt=self.opt,
                    edge_index=self.edge_index,
                    edge_attr=self.edge_attr,
                    n_nodes=self.n_nodes,
                    device=self.device,
                )
                for _ in range(self.n_blocks_dec)
            ]
        )
        self.transform_attention = TransformAttention(
            d_hidden=self.d_hidden, n_heads=self.n_heads, bn_decay=self.bn_decay
        )

        self.decoder = nn.ModuleList(
            [
                SpatioTemporalAttention(
                    n_heads=self.n_heads, d_hidden=self.d_hidden, bn_decay=self.bn_decay
                )
                for _ in range(self.n_blocks_enc)
            ]
        )

        self.fc_in = FullyConnected(
            in_features=[1, self.d_hidden],
            out_features=[self.d_hidden, self.d_hidden],
            activations=[f.softplus, None],
            bn_decay=self.bn_decay,
        )

        self.fc_out = FullyConnected(
            in_features=[self.d_hidden, self.d_hidden],
            out_features=[self.d_hidden, 1],
            activations=[f.softplus, None],
            bn_decay=self.bn_decay,
        )

    def forward(
        self,
        x_signal: torch.Tensor,
        x_temporal: torch.Tensor,
        y_temporal: torch.Tensor,
        store,
    ) -> Tensor:
        # x_signal: (batch_size, n_previous, n_nodes, 1)
        # x_temporal: (batch_size, n_previous, n_nodes, 2)
        # y_temporal: (batch_size, n_future, n_nodes, 2)

        # x (batch_size, n_previous, n_nodes, d_hidden)
        x = self.fc_in(x_signal)

        # temporal_features (batch_size, n_previous+n_future, n_nodes, 2)
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

        x = torch.squeeze(self.fc_out(x), dim=3)
        return x

    def __repr__(self):
        return self.__class__.__name__

    def reset_cached_states(self):
        for module in self.modules():
            if isinstance(module, SpatialDiffusionOde):
                module.cached_encoder_states = []

    def get_cached_states(self):
        cached_states = []
        for module in self.modules():
            if isinstance(module, SpatialDiffusionOde):
                cached_states.extend(module.cached_encoder_states)
        return cached_states
