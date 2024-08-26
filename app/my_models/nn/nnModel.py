import torch
import torch.nn as nn
import torch.functional as F

class MyNNModel(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(in_features = 208*81, out_features = 2000) #208*81, 10
        )
        self.hidden = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features = 2000, out_features = 2000), # 10, 10
            nn.ReLU(),
            nn.Dropout(p=0.5)
        ) for i in range(20)])
        
        self.output = nn.Sequential(
            nn.Linear(in_features = 2000, out_features = 1), # 10, 1
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input(x)
        for module in self.hidden:
            x = module(x)
        x = self.output(x)
        return x
