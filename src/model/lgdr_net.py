from typing import Any, List, Optional

import torch
import torch.nn.functional as f
import torch_geometric
from einops import rearrange, repeat
from torch.nn import GRU

from src.model.base_net import BaseGNN
from src.model.ode.blocks import get_ode_block
from src.model.ode.blocks.funcs import get_ode_function


class LatentGraphDiffusionRecurrentNet(BaseGNN):

    def __init__(self,
                 opt: dict,
                 dataset: torch_geometric.data.Dataset,
                 device):
        super(LatentGraphDiffusionRecurrentNet, self).__init__(opt=opt)
        self.device = device
        self.edge_index, self.edge_attr = dataset.get_adjacency_matrix()

        self.n_previous_steps = opt['n_previous_steps']
        self.n_future_steps = opt['n_future_steps']

        in_memory = opt['in_memory']

        self.n_nodes = dataset[0][0].size()[1] if in_memory else dataset[0].x.size()[1]

        super(LatentGraphDiffusionRecurrentNet, self).__init__(opt=opt)

        self.function = get_ode_function(opt)

        time_tensor = torch.tensor([0, self.T])

        # This is implemented in the datasets that we can choose from so it's safe

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
            # x (batch_size, n_nodes, d_hidden)
            x = rearrange(x, 'b n h -> b h n')
            x = rearrange(self.bn_in(x), 'b h n -> b n h')

        if self.opt['use_augmentation']:
            c_aux = torch.zeros(x.shape).cuda()
            # x (batch_size, n_nodes, d_hidden * 2)
            x = torch.cat([x, c_aux], dim=2)

        self.ode_block.set_x0(x)

        if self.training and self.ode_block.n_reg > 0:
            z, self.reg_states = self.ode_block(x)
        else:
            z = self.ode_block(x)

        if self.opt['use_augmentation']:
            # z (batch_size, n_nodes, d_hidden)
            z = torch.split(z, x.shape[2] // 2, dim=2)[0]

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
