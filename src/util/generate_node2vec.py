import argparse
import os
from pathlib import Path

import torch.cuda
from torch_geometric.datasets import MetrLaInMemory
from torch_geometric.nn import Node2Vec


class Node2VecEmbedder:
    def __init__(self,
                 edge_index: torch.Tensor,
                 embedding_dim: int,
                 walk_length: int,
                 context_size: int,
                 walks_per_node: int,
                 num_negative_samples: int,
                 p,
                 q,
                 n_epochs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Node2Vec(edge_index=edge_index,
                              embedding_dim=embedding_dim,
                              walk_length=walk_length,
                              context_size=context_size,
                              walks_per_node=walks_per_node,
                              num_negative_samples=num_negative_samples,
                              p=p,
                              q=q,
                              sparse=True).to(self.device)

        self.dataloader = self.model.loader(batch_size=32, shuffle=True, num_workers=4)
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)

        self.n_epochs = n_epochs

    def train_step(self):
        self.model.train()
        epoch_loss = 0
        for pos_rw, neg_rw in self.dataloader:
            pos_rw, neg_rw = pos_rw.to(self.device), neg_rw.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw, neg_rw)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.dataloader)

    def generate_embeddings(self):
        for epoch in range(self.n_epochs):
            train_loss = self.train_step()

        return self.model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='Root path of the dataset.')
    parser.add_argument('--inmemory', action='store_true')
    parser.add_argument('--name', type=str, help='Name of the dataset (i.e. metr_la)')
    parser.add_argument('--aug', action='store_true')
    cfg = vars(parser.parse_args())

    # todo - this is hardcoded
    data_root = Path(cfg['root'])
    dataset_name = cfg['name']
    add_temporal_features = cfg['aug']
    tag = f"{dataset_name}" if not add_temporal_features else f"{dataset_name}_aug"
    if cfg['inmemory']:
        data_path = data_root / 'data' / 'inmemory' / tag
        dataset = MetrLaInMemory(root=str(data_path.absolute()),
                                 n_previous_steps=12,
                                 n_future_steps=12)
    else:
        data_path = data_root / 'data' / 'disk' / tag
        dataset = MetrLaInMemory(root=str(data_path.absolute()),
                                 n_previous_steps=12,
                                 n_future_steps=12)

    edge_index = dataset.edge_index

    generator = Node2VecEmbedder(edge_index=edge_index,
                                 embedding_dim=64,
                                 walk_length=20,
                                 context_size=16,
                                 walks_per_node=16,
                                 num_negative_samples=1,
                                 p=1,
                                 q=1,
                                 n_epochs=50)

    embeddings = generator.generate_embeddings()
    if cfg['inmemory']:
        output_path = data_root / 'data' / 'inmemory' / tag
    else:
        output_path = data_root / 'data' / 'disk' / tag

    torch.save(embeddings, os.path.join(output_path, "positional_embeddings.pt"))
