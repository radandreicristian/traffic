from typing import Optional, List, Any

import torch
import torch.nn.functional as f
import torch_geometric
from einops import rearrange
from torch.nn import GRUCell

from src.model.base_net import BaseGNN
from src.model.ode.blocks import get_ode_block
from src.model.ode.blocks.funcs import get_ode_function
from src.util.utils import get_number_of_nodes


class GraphDiffusionRecurrentNet(BaseGNN):
    def __init__(self, opt: dict, dataset: torch_geometric.data.Dataset, device):
        self.device = device
        self.edge_index, self.edge_attr = dataset.get_adjacency_matrix()

        self.n_previous_steps = opt["n_previous_steps"]
        self.n_future_steps = opt["n_future_steps"]

        self.n_nodes = get_number_of_nodes(dataset=dataset, opt=opt)

        super(GraphDiffusionRecurrentNet, self).__init__(opt=opt)

        self.function = get_ode_function(opt)

        time_tensor = torch.tensor([0, self.T])

        # This is implemented in the datasets that we can choose from so it's safe

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

        self.gru_cell = GRUCell(input_size=self.d_hidden, hidden_size=self.d_hidden)

        self.reg_states: Optional[List[Any]] = None

    def forward(
        self, x: torch.Tensor, pos_embedding: torch.Tensor = None
    ) -> torch.Tensor:

        # x (batch_size, n_previous, n_nodes, n_features=1)

        batch_size, n_previous, n_nodes, n_features = x.shape
        y = torch.zeros(batch_size, self.n_future_steps, n_nodes).to(self.device)

        if self.opt["use_beltrami"]:
            raise NotImplementedError("Beltrami not yet implemented.")
        else:
            x = f.dropout(x, self.opt["p_dropout_in"], training=self.training)
            # x (batch_size, n_previous=12, n_nodes=207, n_features=1)

        # x (batch_size, n_previous, n_nodes, d_hidden)
        x = self.fc_in(x)

        if self.opt["use_mlp_in"]:
            x = f.dropout(x, self.opt["p_dropout_model"], training=self.training)
            for layer in self.mlp_in:
                x = layer(x)

        x = rearrange(x, "batch time nodes hid -> time (batch nodes) hid")

        h_t = None

        # Take the input through the
        for i in range(self.n_previous_steps):
            x_t = x[i, :, :]

            if h_t is not None:
                h_t = rearrange(
                    tensor=h_t, pattern="batch nodes hid -> (batch nodes) hid"
                )
                h_t = self.gru_cell(x_t, h_t)
            else:
                h_t = self.gru_cell(x_t)

            h_t = rearrange(
                tensor=h_t,
                pattern="(batch nodes) hid -> batch nodes hid",
                nodes=self.n_nodes,
            )

            if self.opt["use_batch_norm"]:
                # h (batch_size, n_nodes, d_hidden)
                h_t = self.rearrange_batch_norm(h_t)

            if self.opt["use_augmentation"]:
                h_t = self.augment_up(h_t)

            self.ode_block.set_x0(h_t)

            if self.training and self.ode_block.n_reg > 0:
                h_t, self.reg_states = self.ode_block(h_t)
            else:
                h_t = self.ode_block(h_t)

            if self.opt["use_augmentation"]:
                h_t = self.augment_down(h_t)

        # h_t stores the last hidden state of the input (batch, nodes hidden)

        x_t = rearrange(
            tensor=h_t, pattern="batch nodes hidden -> (batch nodes) hidden"
        )
        h_t = None

        for i in range(self.n_future_steps):

            if h_t is not None:
                h_t = rearrange(
                    tensor=h_t, pattern="batch nodes hidden -> (batch nodes) hidden"
                )
                h_t = self.gru_cell(x_t, h_t)
            else:
                h_t = self.gru_cell(x_t)

            h_t = rearrange(
                tensor=h_t,
                pattern="(batch nodes) hid -> batch nodes hid",
                nodes=self.n_nodes,
            )

            if self.opt["use_batch_norm"]:
                # h (batch_size, n_nodes, d_hidden)
                h_t = self.rearrange_batch_norm(h_t)

            if self.opt["use_augmentation"]:
                h_t = self.augment_up(h_t)

            self.ode_block.set_x0(h_t)

            if self.training and self.ode_block.n_reg > 0:
                h_t, self.reg_states = self.ode_block(h_t)
            else:
                h_t = self.ode_block(h_t)

            if self.opt["use_augmentation"]:
                h_t = self.augment_down(h_t)

            o_t = f.relu(h_t)

            if self.opt["use_mlp_out"]:
                for layer in self.mlp_out:
                    o_t = layer(o_t)

            y[:, i, :] = torch.squeeze(self.regressor(o_t))

            x_t = rearrange(o_t, "batch nodes hid -> (batch nodes) hid")

        return y
