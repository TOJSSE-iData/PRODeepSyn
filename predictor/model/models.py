import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        feat = torch.cat([drug1_feat, drug2_feat, cell_feat], 1)
        out = self.network(feat)
        return out
