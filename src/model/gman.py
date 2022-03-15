"""Source:
https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/attention/gman.py"""
from typing import Tuple, List, Union, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_geometric.data
from einops import rearrange, repeat
from torch import Tensor


class Conv2D(nn.Module):
    """An implementation of the 2D-convolution block.

    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_"""

    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 kernel_size: Union[Tuple, List],
                 stride: Union[Tuple, List] = (1, 1),
                 use_bias: bool = True,
                 activation: Optional[Callable[[Tensor], Tensor]] = f.relu,
                 bn_decay: Optional[float] = None) -> None:
        """
        Instantiate the Conv2D object.
        
        :param input_dims: Features in the input (last axis).
        :param output_dims: Resulting features in the output (channels).
        :param kernel_size: Size of the conv kernel.
        :param stride: Stride of the conv kernel (default (1, 1)).
        :param use_bias: Whether to use bias, default True.
        :param activation: Activation function, default is ReLU.
        :param bn_decay: Batch norm decay, default is None.
        """
        super(Conv2D, self).__init__()
        self._activation = activation
        self._conv2d = nn.Conv2d(
            input_dims,
            output_dims,
            kernel_size,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        self._batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self._conv2d.weight)

        if use_bias:
            torch.nn.init.zeros_(self._conv2d.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Conv2D layer.

        :param x: A tensor of shape (batch_size, seq_len, n_nodes, n_features)
        :return: A tensor of the same shape, convoluted.
        """
        x = rearrange(x, 'b l n f -> b f n l')

        # Apply the 2D convolution over the node and 
        x = self._conv2d(x)
        x = self._batch_norm(x)
        if self._activation is not None:
            x = self._activation(x)
        return rearrange(x, 'b f n l -> b l n f')


class FullyConnected(nn.Module):
    """An implementation of a fully-connected convolutional layer.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_
    Args:
        input_dims (int or list): Dimension(s) of input.
        units (int or list): Dimension(s) of outputs in each 2D convolution block.
        activations (Callable or list): Activation function(s).
        bn_decay (float, optional): Batch normalization momentum, default is None.
        use_bias (bool, optional): Whether to use bias, default is True.
    """

    def __init__(
            self,
            input_dims: Union[int, List[int]],
            units: Union[int, List[int]],
            activations: Optional[Union[Callable[[Tensor], Tensor], List[Callable[[Tensor], Tensor]]]],
            bn_decay: float,
            use_bias: bool = True) -> None:
        """
        Instantiate an FullyConnected layer.
        :param input_dims: Dimension(s) of the input.
        :param units: Dimension(s) of the outputs in each 2D conv block.
        :param activations: Activation function(s).
        :param bn_decay: Batch norm decay, default is None.
        :param use_bias: Whether to use bias, default True.
        """
        super(FullyConnected, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        assert type(units) == list
        self._conv2ds = nn.ModuleList(
            [Conv2D(
                input_dims=input_dim,
                output_dims=num_unit,
                kernel_size=[1, 1],
                stride=[1, 1],
                use_bias=use_bias,
                activation=activation,
                bn_decay=bn_decay) for input_dim, num_unit, activation in zip(input_dims, units, activations)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Making a forward pass of the fully-connected layer.
        Arg types:
            * **X** (PyTorch Float Tensor) - Input tensor, with shape (batch_size, num_his, num_nodes, 1).
        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, num_his, num_nodes, units[-1]).
        """
        for conv in self._conv2ds:
            x = conv(x)
        return x


class SpatioTemporalEmbedding(nn.Module):
    def __init__(self,
                 d_hidden: int,
                 bn_decay: float,
                 use_bias: bool = True):
        super(SpatioTemporalEmbedding, self).__init__()
        self.fully_connected_spatial = FullyConnected(input_dims=[d_hidden, d_hidden],
                                                      units=[d_hidden, d_hidden],
                                                      activations=[f.relu, None],
                                                      bn_decay=bn_decay,
                                                      use_bias=use_bias)
        self.fully_connected_temporal = FullyConnected(input_dims=[2, d_hidden],
                                                       units=[d_hidden, d_hidden],
                                                       activations=[f.relu, None],
                                                       bn_decay=bn_decay,
                                                       use_bias=use_bias)

    def forward(self,
                spatial_embeddings: Tensor,
                temporal_embeddings: Tensor) -> Tensor:
        """

        :param spatial_embeddings: Spatial embedding tensor of shape (n_nodes, d_hidden)
        :param temporal_embeddings: Temporal embedding tensor of shape (batch_size, seq_len, n_nodes, 2)
        :return: Spatio-temporal embedding, of shape (batch_size, seq_len, n_nodes, d_hidden).
        """

        # spatial_embeddings (n_nodes, d_hidden)
        spatial_embeddings = torch.unsqueeze(torch.unsqueeze(spatial_embeddings, dim=0), dim=0)
        spatial_embeddings = self.fully_connected_spatial(spatial_embeddings)

        # temporal_embeddings (batch_size, seq_len, n_nodes, d_hidden)
        temporal_embeddings = self.fully_connected_temporal(temporal_embeddings)

        # automatic broadcast to (batch_size, seq_len, n_nodes, d_hidden)
        return spatial_embeddings + temporal_embeddings


class SpatialAttention(nn.Module):
    def __init__(self,
                 d_hidden,
                 n_heads,
                 bn_decay) -> None:
        super(SpatialAttention, self).__init__()
        assert d_hidden % n_heads == 0, "Hidden size not divisible by number of heads."

        self.d_head = d_hidden // n_heads
        self.n_heads = n_heads

        self.fully_connected_q = FullyConnected(input_dims=2 * d_hidden,
                                                units=d_hidden,
                                                activations=f.relu,
                                                bn_decay=bn_decay)

        self.fully_connected_k = FullyConnected(input_dims=2 * d_hidden,
                                                units=d_hidden,
                                                activations=f.relu,
                                                bn_decay=bn_decay)

        self.fully_connected_v = FullyConnected(input_dims=2 * d_hidden,
                                                units=d_hidden,
                                                activations=f.relu,
                                                bn_decay=bn_decay)
        self.fully_connected_out = FullyConnected(input_dims=d_hidden,
                                                  units=d_hidden,
                                                  activations=f.relu,
                                                  bn_decay=bn_decay)

    def forward(self, x, ste):
        batch_size, _, _, _ = x.shape
        x = torch.cat((x, ste), dim=-1)
        queries = torch.cat(torch.split(self.fully_connected_q(x), self.d_head, dim=-1), dim=0)
        keys = torch.cat(torch.split(self.fully_connected_k(x), self.d_head, dim=-1), dim=0)
        values = torch.cat(torch.split(self.fully_connected_v(x), self.d_head, dim=-1), dim=0)

        keys = rearrange(keys, 'bxn_heads seq_len n_nodes d_head -> bxn_heads seq_len d_head n_nodes')

        attention_scores = f.softmax(queries @ keys / self.d_head ** .5, dim=-1)
        attention = torch.cat(torch.split(attention_scores @ values, batch_size, dim=0), dim=-1)

        del queries, keys, values, attention_scores
        return self.fully_connected_out(attention)


class TemporalAttention(nn.Module):
    def __init__(self,
                 d_hidden,
                 n_heads,
                 bn_decay,
                 use_mask: bool = True) -> None:
        super(TemporalAttention, self).__init__()
        assert d_hidden % n_heads == 0, "Hidden size not divisible by number of heads."

        self.d_head = d_hidden // n_heads
        self.n_heads = n_heads
        self.use_mask = use_mask
        self.fully_connected_q = FullyConnected(input_dims=2 * d_hidden,
                                                units=d_hidden,
                                                activations=f.relu,
                                                bn_decay=bn_decay)

        self.fully_connected_k = FullyConnected(input_dims=2 * d_hidden,
                                                units=d_hidden,
                                                activations=f.relu,
                                                bn_decay=bn_decay)

        self.fully_connected_v = FullyConnected(input_dims=2 * d_hidden,
                                                units=d_hidden,
                                                activations=f.relu,
                                                bn_decay=bn_decay)
        self.fully_connected_out = FullyConnected(input_dims=d_hidden,
                                                  units=d_hidden,
                                                  activations=f.relu,
                                                  bn_decay=bn_decay)

    def forward(self, x, ste) -> Tensor:
        batch_size, seq_len, n_nodes, _ = x.shape
        x = torch.cat((x, ste), dim=-1)
        queries = torch.cat(torch.split(self.fully_connected_q(x), self.d_head, dim=-1), dim=0)
        keys = torch.cat(torch.split(self.fully_connected_k(x), self.d_head, dim=-1), dim=0)
        values = torch.cat(torch.split(self.fully_connected_v(x), self.d_head, dim=-1), dim=0)

        queries = rearrange(queries, 'bxn_heads seq_len n_nodes d_head -> bxn_heads n_nodes seq_len d_head')
        keys = rearrange(keys, 'bxn_heads seq_len n_nodes d_head -> bxn_heads n_nodes d_head seq_len')
        values = rearrange(values, 'bxn_heads seq_len n_nodes d_head -> bxn_heads n_nodes seq_len d_head')

        attention_scores = queries @ keys / self.d_head ** .5
        if self.use_mask:
            mask = torch.tril(torch.ones(seq_len, seq_len).to(x.device))
            mask = repeat(mask, 'a b -> j n a b', j=batch_size * self.n_heads, n=n_nodes).to(torch.bool)
            condition = torch.tensor([-(2 ** 15) + 1], dtype=torch.float32).to(x.device)
            attention_scores = torch.where(mask, attention_scores, condition)
        attention_scores = f.softmax(attention_scores, dim=-1)
        attention = torch.matmul(attention_scores, values)
        attention = rearrange(attention, 'b n l d -> b l n d')
        attention = torch.cat(torch.split(attention, batch_size, dim=0), -1)

        del queries, keys, values, attention_scores
        return self.fully_connected_out(attention)


class GatedFusion(nn.Module):
    def __init__(self, d_hidden, bn_decay):
        super(GatedFusion, self).__init__()
        self.fully_connected_spatial = FullyConnected(input_dims=d_hidden,
                                                      units=d_hidden,
                                                      activations=None,
                                                      bn_decay=bn_decay,
                                                      use_bias=True)
        self.fully_connected_temporal = FullyConnected(input_dims=d_hidden,
                                                       units=d_hidden,
                                                       activations=None,
                                                       bn_decay=bn_decay,
                                                       use_bias=True)

        self.fully_connected_out = FullyConnected(input_dims=[d_hidden, d_hidden],
                                                  units=[d_hidden, d_hidden],
                                                  activations=[f.relu, None],
                                                  bn_decay=bn_decay)

    def forward(self,
                h_spatial,
                h_temporal) -> None:
        h_spatial = self.fully_connected_spatial(h_spatial)
        h_temporal = self.fully_connected_temporal(h_temporal)
        z = torch.sigmoid(h_temporal + h_spatial)
        h = torch.mul(z, h_spatial) + torch.mul(1 - z, h_temporal)
        h = self.fully_connected_out(h)

        del h_spatial, h_temporal, z
        return h


class SpatioTemporalAttention(nn.Module):
    def __init__(self, n_heads, d_hidden, bn_decay):
        super(SpatioTemporalAttention, self).__init__()

        self.spatial_attention = SpatialAttention(d_hidden=d_hidden, n_heads=n_heads, bn_decay=bn_decay)
        self.temporal_attention = TemporalAttention(d_hidden=d_hidden, n_heads=n_heads, bn_decay=bn_decay)
        self.gated_fusion = GatedFusion(d_hidden=d_hidden, bn_decay=bn_decay)

    def forward(self, x, ste):
        h_spatial = self.spatial_attention(x, ste)
        h_temporal = self.temporal_attention(x, ste)
        h = self.gated_fusion(h_spatial, h_temporal)

        del h_spatial, h_temporal
        return x + h


class TransformAttention(nn.Module):
    def __init__(self,
                 d_hidden,
                 n_heads,
                 bn_decay):
        super(TransformAttention, self).__init__()
        assert d_hidden % n_heads == 0, "Hidden size not divisible by number of heads."
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads
        self.fc_query = FullyConnected(input_dims=d_hidden,
                                       units=d_hidden,
                                       activations=f.relu,
                                       bn_decay=bn_decay)

        self.fc_key = FullyConnected(input_dims=d_hidden,
                                     units=d_hidden,
                                     activations=f.relu,
                                     bn_decay=bn_decay)

        self.fc_value = FullyConnected(input_dims=d_hidden,
                                       units=d_hidden,
                                       activations=f.relu,
                                       bn_decay=bn_decay)
        self.fully_connected_out = FullyConnected(input_dims=d_hidden,
                                                  units=d_hidden,
                                                  activations=f.relu,
                                                  bn_decay=bn_decay)

    def forward(self,
                x,
                ste_previous,
                ste_future) -> torch.Tensor:
        batch_size, _, _, _ = x.shape
        queries = torch.cat(torch.split(self.fc_query(ste_future), self.n_heads, dim=-1), dim=0)
        keys = torch.cat(torch.split(self.fc_key(ste_previous), self.n_heads, dim=-1), dim=0)
        values = torch.cat(torch.split(self.fc_value(x), self.n_heads, dim=-1), dim=0)

        queries = rearrange(queries, 'bxn_heads seq_len n_nodes d_head -> bxn_heads n_nodes seq_len d_head')
        keys = rearrange(keys, 'bxn_heads seq_len n_nodes d_head -> bxn_heads n_nodes d_head seq_len')
        values = rearrange(values, 'bxn_heads seq_len n_nodes d_head -> bxn_heads n_nodes seq_len d_head')
        attention_scores = f.softmax(queries @ keys / self.d_head ** .5, dim=-1)
        attention = rearrange(attention_scores @ values, 'bk n l dh -> bk l n dh')

        del keys, queries, values, attention_scores
        return self.fully_connected_out(torch.cat(torch.split(attention, batch_size, dim=0), dim=-1))


class GraphMultiAttentionNet(nn.Module):
    def __init__(self,
                 opt: dict,
                 dataset: torch_geometric.data.Dataset,
                 device: torch.device):
        super(GraphMultiAttentionNet, self).__init__()

        self.device = device
        self.d_hidden = opt.get('d_hidden')
        self.n_heads = opt.get('n_heads')
        self.bn_decay = opt.get('bn_decay', 0.9)

        self.n_previous = opt.get('n_previous_steps')
        self.n_future = opt.get('n_future_steps')
        self.n_blocks = opt.get('n_blocks')

        # Todo - This will require grad later)
        self.positional_embeddings = nn.Parameter(data=opt.get('positional_embeddings'), requires_grad=False)
        self.st_embedding = SpatioTemporalEmbedding(d_hidden=self.d_hidden,
                                                    bn_decay=self.bn_decay)

        self.encoder = nn.ModuleList([SpatioTemporalAttention(n_heads=self.n_heads,
                                                              d_hidden=self.d_hidden,
                                                              bn_decay=self.bn_decay) for _ in range(self.n_blocks)])
        self.transform_attention = TransformAttention(d_hidden=self.d_hidden,
                                                      n_heads=self.n_heads,
                                                      bn_decay=self.bn_decay)
        self.decoder = nn.ModuleList([SpatioTemporalAttention(n_heads=self.n_heads,
                                                              d_hidden=self.d_hidden,
                                                              bn_decay=self.bn_decay) for _ in range(self.n_blocks)])

        self.fc_in = FullyConnected(input_dims=[1, self.d_hidden],
                                    units=[self.d_hidden, self.d_hidden],
                                    activations=[f.relu, None],
                                    bn_decay=self.bn_decay)

        self.fc_out = FullyConnected(input_dims=[self.d_hidden, self.d_hidden],
                                     units=[self.d_hidden, 1],
                                     activations=[f.relu, None],
                                     bn_decay=self.bn_decay)

    def forward(self,
                x_signal: torch.Tensor,
                x_temporal: torch.Tensor,
                y_temporal: torch.Tensor) -> Tensor:
        # x_signal: (batch_size, n_previous, n_nodes)
        # x_temporal: (batch_size, n_previous, n_nodes, 2)
        # y_temporal: (batch_size, n_future, n_nodes, 2)

        x = self.fc_in(x_signal)

        first_future_index = x_temporal.shape[1]

        # temporal_features (batch_size, n_previous+n_future, n_nodes, 2)
        temporal_features = torch.cat((x_temporal, y_temporal), dim=1)

        st_embeddings = self.st_embedding(spatial_embeddings=self.positional_embeddings,
                                          temporal_embeddings=temporal_features)

        st_embeddings_previous = st_embeddings[:, :first_future_index, ...]
        st_embeddings_future = st_embeddings[:, first_future_index:, ...]

        for block in self.encoder:
            x = block(x, st_embeddings_previous)

        x = self.transform_attention(x, st_embeddings_previous, st_embeddings_future)

        for block in self.encoder:
            x = block(x, st_embeddings_future)

        x = torch.squeeze(self.fc_out(x), 3)
        return x
