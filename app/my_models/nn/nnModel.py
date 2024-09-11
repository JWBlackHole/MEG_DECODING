import torch
import torch.nn as nn
import torch.functional as F

class MyNNModel(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(in_features = 3, out_features = 10),
            nn.ReLU(inplace = True)
        )
        self.hidden = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features = 10, out_features = 10),
            nn.ReLU(inplace = True)
        ) for i in range(3)])
        
        self.output = nn.Sequential(
            nn.Linear(in_features = 10, out_features = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input(x)
        for module in self.hidden:
            x = module(x)
        x = self.output(x)
        return x