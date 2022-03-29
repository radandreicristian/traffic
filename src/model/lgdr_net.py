from typing import Any, List, Optional

import torch
import torch.nn.functional as f
import torch_geometric
from einops import rearrange, repeat
from torch.nn import GRU

from src.model.base_net import BaseGNN
from src.model.ode.blocks import get_ode_block
from src.model.ode.blocks.funcs import get_ode_function
from src.util.utils import get_number_of_nodes

class LatentGraphDiffusionRecurrentNet(BaseGNN):
    """
    A latent graph diffusion recurrent net. There is a single diffusion layer, between a recurrent encoder and a
    recurrent decoder.
    """

    def __init__(self,
                 opt: dict,
                 dataset: torch_geometric.data.Dataset,
                 device: torch.device) -> None:
        self.device = device
        self.edge_index, self.edge_attr = dataset.get_adjacency_matrix()

        self.n_previous_steps = opt['n_previous_steps']
        self.n_future_steps = opt['n_future_steps']

        self.n_nodes = get_number_of_nodes(dataset=dataset, opt=opt)

        super(LatentGraphDiffusionRecurrentNet, self).__init__(opt=opt)

        self.function = get_ode_function(opt)

        time_tensor = torch.tensor([0, self.T])

        self.ode_block = get_ode_block(opt)(ode_func=self.function,
                                            reg_funcs=self.reg_funcs,
                                            opt=opt,
                                            edge_index=self.edge_index,
                                            edge_attr=self.edge_attr,
                                            n_nodes=self.n_nodes,
                                            t=time_tensor,
                                            device=self.device)

        self.rnn = GRU(input_size=self.d_hidden,
                       hidden_size=self.d_hidden,
                       num_layers=1,
                       batch_first=True)

        self.reg_states: Optional[List[Any]] = None

    def forward(self,
                x: torch.Tensor,
                pos_embedding: torch.Tensor = None) -> torch.Tensor:

        # x (batch_size, n_previous, n_nodes, n_features=1)

        if self.opt['use_beltrami']:
            raise NotImplementedError('Beltrami not yet implemented.')
        else:
            x = f.dropout(x, self.opt['p_dropout_in'], training=self.training)

        # x (batch_size, n_previous, n_nodes, d_hidden)
        x = self.fc_in(x)

        if self.opt['use_mlp_in']:
            x = f.dropout(x, self.opt['p_dropout_model'], training=self.training)
            for layer in self.mlp_in:
                # x (batch_size, n_previous, n_nodes, d_hidden)
                x = f.dropout(x + layer(f.relu(x)), self.opt['p_dropout_model'], training=self.training)

        # x ((batch_size, n_nodes), n_previous, d_hidden) - Prepare shape for RNN (req. 3D)
        x = rearrange(x, 'b p n h -> (b n) p h')

        # x ((batch_size, n_nodes), n_previous, d_hidden)
        x, _ = self.rnn(x)

        # x (batch_size, n_previous, n_nodes, d_hidden) - Rearrange back into a 4D tensor
        x = rearrange(x, '(b n) p h -> b p n h', n=self.n_nodes)

        # x (batch_size, n_nodes, d_hidden) - Take only the last output state
        x = x[:, -1, :, :]

        if self.opt['use_batch_norm']:
            x = self.rearrange_batch_norm(x)

        if self.opt['use_augmentation']:
            x = self.augment_up(x)

        self.ode_block.set_x0(x)

        if self.training and self.ode_block.n_reg > 0:
            z, self.reg_states = self.ode_block(x)
        else:
            z = self.ode_block(x)

        if self.opt['use_augmentation']:
            z = self.augment_down(z)

        z = f.relu(z)

        if self.opt['use_mlp_out']:
            for layer in self.mlp_out:
                # z (batch_size, n_nodes, d_hidden)
                z = f.relu(layer(z))

        z = f.dropout(z, self.opt['p_dropout_model'])

        # z (batch_size, n_future_steps, n_nodes, n_features)
        z = repeat(z, 'b n h -> b f n h', f=self.n_future_steps)

        z = rearrange(z, 'b f n h -> (b n) f h')

        # z (batch_size, n_future_steps, n_nodes, n_features)
        z, _ = self.rnn(z)

        # z (batch_Size, n_future_steps, n_node, d_hidden)
        z = rearrange(z, '(b n) f h -> b f n h', n=self.n_nodes)

        # z (batch_size, n_nodes, n_future) - Per node final prediction
        z = torch.squeeze(self.regressor(z))

        del x
        torch.cuda.empty_cache()
        # z = rearrange(z, 'b n f -> b f n')

        return z
