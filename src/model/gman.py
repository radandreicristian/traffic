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
    """An implementation of a 2D-convolution block, with batch normalization on top."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 kernel_size: Union[Tuple, List],
                 stride: Union[Tuple, List],
                 use_bias: bool = True,
                 activation: Optional[Callable[[Tensor], Tensor]] = f.softplus,
                 bn_decay: Optional[float] = None) -> None:
        """
        Instantiate the ChainConv2D object.

        :param in_features: Features in the input (last axis).
        :param out_features: Resulting features in the output (channels).
        :param kernel_size: Size of the conv kernel.
        :param stride: Stride of the conv kernel (default (1, 1)).
        :param use_bias: Whether to use bias, default True.
        :param activation: Activation function, default is softPlus (works better for ODEs than ReLU variations).
        :param bn_decay: Batch norm decay, default is None.
        """
        super(Conv2D, self).__init__()
        self.activation = activation
        self.conv2d = nn.Conv2d(
            in_features,
            out_features,
            kernel_size,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        self.batch_norm = nn.BatchNorm2d(out_features, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv2d.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv2d.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Conv2D layer.

        :param x: A tensor of shape (batch_size, seq_len, n_nodes, n_features)
        :return: A tensor of the same shape, convoluted.
        """
        x = rearrange(x, 'batch seq_len n_nodes n_features -> batch n_features n_nodes seq_len')

        # Apply the 2D convolution over dims 0 and 1 (number of features = number of channels)
        x = self.conv2d(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return rearrange(x, 'b f n l -> b l n f')


class FullyConnected(nn.Module):
    """An implementation of a fully-connected convolutional layer.

    As from the GMAN original implementation, the FC layer is a 2D conv over a general input of shape
    (batch_size, seq_len, n_nodes, n_features). The layer contains sequential convolution blocks, described by the
    parameters. The parameters of the function are expected to be either single values (ints/tuples), case in which
    the ChainConv2D will contain a single conv layer, or lists of equal sizes of int/tuple, case in which the lists
    are zipped and each resulting tuple makes a convolutional layer.
    """

    def __init__(
            self,
            in_features: Union[int, List[int]],
            out_features: Union[int, List[int]],
            activations: Optional[Union[Callable[[Tensor], Tensor], List[Callable[[Tensor], Tensor]]]],
            bn_decay: float,
            kernel_size: Union[Tuple, List] = (1, 1),
            stride: Union[Tuple, List] = (1, 1),
            use_bias: bool = True) -> None:
        """
        Instantiate an FullyConnected layer.
        :param in_features: Dimension(s) of the inputs in each 2D conv block.
        :param out_features: Dimension(s) of the outputs in each 2D conv block.
        :param activations: Activation function(s).
        :param bn_decay: Batch norm decay, default is None.
        :param use_bias: Whether to use bias, default True.
        """
        super(FullyConnected, self).__init__()
        if isinstance(out_features, int):
            out_features = [out_features]
            in_features = [in_features]
            activations = [activations]
        assert type(out_features) == list
        self.conv2ds = nn.ModuleList(
            [Conv2D(
                in_features=in_features,
                out_features=out_features,
                kernel_size=kernel_size,
                stride=stride,
                use_bias=use_bias,
                activation=activation,
                bn_decay=bn_decay) for in_features, out_features, activation in
                zip(in_features, out_features, activations)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forwards the input through the fully-connected layer.

        :param x: The input tensor of shape (batch_size, seq_len, n_nodes, n_features)
        :return: The resulting tensor of shape (batch_size, seq_len, n_nodes, n_features')
        """
        for conv in self.conv2ds:
            x = conv(x)
        return x


class SpatioTemporalEmbedding(nn.Module):
    def __init__(self,
                 d_hidden: int,
                 bn_decay: float,
                 use_bias: bool = True):
        super(SpatioTemporalEmbedding, self).__init__()
        self.fully_connected_spatial = FullyConnected(in_features=[d_hidden, d_hidden],
                                                      out_features=[d_hidden, d_hidden],
                                                      activations=[f.relu, None],
                                                      bn_decay=bn_decay,
                                                      use_bias=use_bias)
        self.fully_connected_temporal = FullyConnected(in_features=[2, d_hidden],
                                                       out_features=[d_hidden, d_hidden],
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

        self.fully_connected_q = FullyConnected(in_features=2 * d_hidden,
                                                out_features=d_hidden,
                                                activations=f.relu,
                                                bn_decay=bn_decay)

        self.fully_connected_k = FullyConnected(in_features=2 * d_hidden,
                                                out_features=d_hidden,
                                                activations=f.relu,
                                                bn_decay=bn_decay)

        self.fully_connected_v = FullyConnected(in_features=2 * d_hidden,
                                                out_features=d_hidden,
                                                activations=f.relu,
                                                bn_decay=bn_decay)
        self.fully_connected_out = FullyConnected(in_features=d_hidden,
                                                  out_features=d_hidden,
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
        self.fully_connected_q = FullyConnected(in_features=2 * d_hidden,
                                                out_features=d_hidden,
                                                activations=f.relu,
                                                bn_decay=bn_decay)

        self.fully_connected_k = FullyConnected(in_features=2 * d_hidden,
                                                out_features=d_hidden,
                                                activations=f.relu,
                                                bn_decay=bn_decay)

        self.fully_connected_v = FullyConnected(in_features=2 * d_hidden,
                                                out_features=d_hidden,
                                                activations=f.relu,
                                                bn_decay=bn_decay)
        self.fully_connected_out = FullyConnected(in_features=d_hidden,
                                                  out_features=d_hidden,
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
        self.fully_connected_spatial = FullyConnected(in_features=d_hidden,
                                                      out_features=d_hidden,
                                                      activations=None,
                                                      bn_decay=bn_decay,
                                                      use_bias=True)
        self.fully_connected_temporal = FullyConnected(in_features=d_hidden,
                                                       out_features=d_hidden,
                                                       activations=None,
                                                       bn_decay=bn_decay,
                                                       use_bias=True)

        self.fully_connected_out = FullyConnected(in_features=[d_hidden, d_hidden],
                                                  out_features=[d_hidden, d_hidden],
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
        self.fc_query = FullyConnected(in_features=d_hidden,
                                       out_features=d_hidden,
                                       activations=f.relu,
                                       bn_decay=bn_decay)

        self.fc_key = FullyConnected(in_features=d_hidden,
                                     out_features=d_hidden,
                                     activations=f.relu,
                                     bn_decay=bn_decay)

        self.fc_value = FullyConnected(in_features=d_hidden,
                                       out_features=d_hidden,
                                       activations=f.relu,
                                       bn_decay=bn_decay)
        self.fully_connected_out = FullyConnected(in_features=d_hidden,
                                                  out_features=d_hidden,
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

        self.fc_in = FullyConnected(in_features=[1, self.d_hidden],
                                    out_features=[self.d_hidden, self.d_hidden],
                                    activations=[f.relu, None],
                                    bn_decay=self.bn_decay)

        self.fc_out = FullyConnected(in_features=[self.d_hidden, self.d_hidden],
                                     out_features=[self.d_hidden, 1],
                                     activations=[f.relu, None],
                                     bn_decay=self.bn_decay)

    def forward(self,
                x_signal: torch.Tensor,
                x_temporal: torch.Tensor,
                y_temporal: torch.Tensor) -> Tensor:
        # x_signal: (batch_size, n_previous, n_nodes, 1)
        # x_temporal: (batch_size, n_previous, n_nodes, 2)
        # y_temporal: (batch_size, n_future, n_nodes, 2)

        x = self.fc_in(x_signal)

        # temporal_features (batch_size, n_previous+n_future, n_nodes, 2)
        temporal_features = torch.cat((x_temporal, y_temporal), dim=1)
        first_future_index = temporal_features.shape[1] // 2

        st_embeddings = self.st_embedding(spatial_embeddings=self.positional_embeddings,
                                          temporal_embeddings=temporal_features)

        st_embeddings_previous = st_embeddings[:, :first_future_index, ...]
        st_embeddings_future = st_embeddings[:, first_future_index:, ...]

        for block in self.encoder:
            x = block(x, st_embeddings_previous)

        x = self.transform_attention(x, st_embeddings_previous, st_embeddings_future)

        for block in self.decoder:
            x = block(x, st_embeddings_future)

        x = torch.squeeze(self.fc_out(x), 3)
        return x
