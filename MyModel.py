import torch
import torch.nn as nn
import torch.functional as F

class MyModel(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(in_features = 208*81, out_features = 10)
        )
        self.hidden = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features = 10, out_features = 10)
        ) for i in range(1)])
        
        self.output = nn.Sequential(
            nn.Linear(in_features = 10, out_features = 1)
        )

    def forward(self, x):
        x = self.input(x)
        for module in self.hidden:
            x = module(x)
        x = self.output(x)
        return x
