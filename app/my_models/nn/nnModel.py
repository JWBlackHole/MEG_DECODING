import torch
import torch.nn as nn
import torch.functional as F

class MyNNModel(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(in_features = 208*81, out_features = 5000),
            nn.ReLU(inplace = True)
        )
        self.hidden = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features = 5000, out_features = 1000),
            nn.ReLU(inplace = True)
        ) for i in range(0)])
        
        # self.output = nn.Sequential(
        #     nn.Linear(in_features = 5000, out_features = 1),
        #     nn.ReLU(inplace = True)
        # )
        self.output = nn.Sequential(
            nn.Linear(in_features = 5000, out_features = 1)
        )

    def forward(self, x):
        x = self.input(x)
        for module in self.hidden:
            x = module(x)
        x = self.output(x)
        return x