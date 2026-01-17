import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GraphCNN(torch.nn.Module):
    def __init__(self, num_features):
        super(GraphCNN, self).__init__()

        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 64)

        self.fc1 = Linear(128, 64)
        self.fc2 = Linear(64, 1)  # Binary classification

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph-level pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        x = torch.cat([x_mean, x_max], dim=1)

        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x
