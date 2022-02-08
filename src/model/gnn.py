import torch
import torch.nn.functional as f
import torch_geometric
from einops import rearrange

from src.model.gnn_base import BaseGNN
from src.model.ode.blocks import get_ode_block
from src.model.ode.blocks.funcs import get_ode_function


class GNN(BaseGNN):
    """

    """

    def __init__(self,
                 opt: dict,
                 dataset: torch_geometric.data.Dataset,
                 device):

        self.device = device
        self.edge_index, self.edge_attr = dataset.get_adjacency_matrix()

        self.n_previous_steps = opt['n_previous_steps']
        self.n_future_steps = opt['n_future_steps']

        in_memory = opt['in_memory']

        if in_memory:
            n_nodes = dataset[0][0].size()[1]  # Todo maybe find a better way to do this
        else:
            n_nodes = dataset[0].x.size()[1]

        n_features = opt['n_features']

        super(GNN, self).__init__(opt=opt,
                                  n_features=n_features,
                                  n_nodes=n_nodes)
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

    def forward(self,
                x: torch.Tensor,
                pos_embedding: torch.Tensor = None) -> torch.Tensor:

        # x (batch_size, n_previous=12, n_nodes=207, n_features=1)

        if self.opt['use_beltrami']:
            raise NotImplementedError('Beltrami not yet implemented.')
        else:
            x = f.dropout(x, self.opt['p_dropout_in'], training=self.training)
            # x (batch_size, n_nodes, n_features, d_hidden)

        # x (batch_size, n_nodes, d_hidden, n_previous)

        # x (batch_size, n_previous, n_nodes)
        x = torch.squeeze(x)

        # x (batch_size, n_nodes, n_previous)
        x = rearrange(x, 'b p n -> b n p')

        # x (batch_size, n_nodes, d_hidden)
        x = self.fc_in(x)

        if self.opt['use_mlp_in']:
            x = f.dropout(x, self.opt['p_dropout_model'], training=self.training)
            for layer in self.mlp_in:
                # x (batch_size, n_nodes, d_hidden)
                x = f.dropout(x + layer(f.relu(x)), self.opt['p_dropout_model'], training=self.training)

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

        # z (batch_size, n_nodes, n_future) - Per node final prediction
        z = self.regressor(z)

        del x
        torch.cuda.empty_cache()
        z = rearrange(z, 'b n p -> b p n')

        return z
