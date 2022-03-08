from typing import List, Any, Optional, Union, Tuple

import torch
import torch.nn.functional as f
import torch_geometric.data
from einops import rearrange
from torch import Tensor

from src.model.base_net import BaseGNN
from src.model.ode.blocks import get_ode_block
from src.model.ode.blocks.funcs import get_ode_function


class OdeNet(BaseGNN):

    def __init__(self,
                 opt: dict,
                 dataset: torch_geometric.data.Dataset,
                 device: torch.device) -> None:

        self.device = device
        self.edge_index, self.edge_attr = dataset.get_adjacency_matrix()

        self.n_previous_steps = opt['n_previous_steps']
        self.n_future_steps = opt['n_future_steps']

        in_memory = opt['in_memory']

        self.n_nodes = dataset[0][0].size()[1] if in_memory else dataset[0].x.size()[1]

        super(OdeNet, self).__init__(opt=opt)
        self.return_previous_losses = True

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

        self.reg_states: Optional[List[Any]] = None

    def forward(self,
                x0: Tensor,
                pos_embedding: Tensor = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        if self.opt['use_beltrami']:
            raise NotImplementedError('Beltrami not yet implemented.')
        else:
            x0 = f.dropout(x0, self.opt['p_dropout_in'], training=self.training)

        # x (batch_size, n_previous, n_nodes, d_hidden)
        x0 = self.fc_in(x0)

        if self.opt['use_mlp_in']:
            x0 = f.dropout(x0, self.opt['p_dropout_model'], training=self.training)
            for layer in self.mlp_in:
                # x (batch_size, n_previous, n_nodes, d_hidden)
                x0 = f.dropout(x0 + layer(f.relu(x0)), self.opt['p_dropout_model'], training=self.training)

        x0 = x0[:, 0, ...]

        predicted_previous = []
        predicted_future = []

        for step in range(1, self.n_previous_steps):
            if self.opt['use_batch_norm']:
                x0 = self.rearrange_batch_norm(x0)

            if self.opt['use_augmentation']:
                x0 = self.augment_up(x0)

            self.ode_block.set_x0(x0)

            if self.training and self.ode_block.n_reg > 0:
                z_next, self.reg_states = self.ode_block(x0)
            else:
                z_next = self.ode_block(x0)

            if self.opt['use_augmentation']:
                z_next = self.augment_down(z_next)

            h = z_next
            if self.opt['use_mlp_out']:
                for layer in self.mlp_out:
                    # z (batch_size, n_nodes, d_hidden)
                    h = f.relu(layer(h))

            predicted_previous.append(self.regressor(h))
            x0 = z_next

        for step in range(self.n_future_steps):
            if self.opt['use_batch_norm']:
                x0 = self.rearrange_batch_norm(x0)

            if self.opt['use_augmentation']:
                x0 = self.augment_up(x0)

            self.ode_block.set_x0(x0)

            if self.training and self.ode_block.n_reg > 0:
                z_next, self.reg_states = self.ode_block(x0)
            else:
                z_next = self.ode_block(x0)

            if self.opt['use_augmentation']:
                z_next = self.augment_down(z_next)

            h = z_next
            if self.opt['use_mlp_out']:
                for layer in self.mlp_out:
                    # z (batch_size, n_nodes, d_hidden)
                    h = f.relu(layer(h))

            predicted_future.append(self.regressor(h))
            x0 = z_next

        predicted_future = torch.squeeze(torch.stack(predicted_future, dim=0))
        predicted_future = rearrange(predicted_future, 'f b n -> b f n')

        predicted_previous = torch.squeeze(torch.stack(predicted_previous, dim=0))
        predicted_previous = rearrange(predicted_previous, 'p b n -> b p n')

        if self.return_previous_losses:
            return predicted_previous, predicted_future
        else:
            return predicted_future
