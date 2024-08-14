import torch
import torch.nn as nn
import torch.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class MyModel(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.input: nn.Sequential = nn.Sequential(
            nn.Linear(in_features = 208*81, out_features = 1000)
        )
        
        self.hidden = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features = 1000, out_features = 1000)
        ) for i in range(5)])
        
        self.output = nn.Sequential(
            nn.Linear(in_features = 1000, out_features = 1)
        )
        
        self.input.apply(init_weights)
        for sequential in self.hidden:
            sequential.apply(init_weights)
        self.output.apply(init_weights)

    def forward(self, x):
        x = self.input(x)
        for module in self.hidden:
            x = module(x)
        x = self.output(x)
        return x