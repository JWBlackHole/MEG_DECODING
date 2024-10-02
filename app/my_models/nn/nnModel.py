import torch
import torch.nn as nn
import torch.functional as F

from loguru import logger

class MyNNModel(nn.Module):
    def __init__(self, nchans: int, ntimes:int):
        super().__init__()
        self.nchans = nchans
        self.ntimes = ntimes

        # [notes!]
        # 2nd dimension of x need to = in_features of input!

        self.input = nn.Sequential(
            nn.Linear(in_features = self.nchans * self.ntimes, out_features = 5000),
            nn.GELU()
        )
        
        self.fn1 = nn.Sequential(
            nn.Linear(in_features = 5000, out_features = 2500),
            nn.GELU()
        )
        self.fn2 = nn.Sequential(
            nn.Linear(in_features = 2500, out_features = 1000),
            nn.GELU()
        )
        
        self.output = nn.Sequential(
            nn.Linear(in_features = 1000, out_features = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        # print(f"initial x shape: {x.shape}")
        # logger.debug(f"initial x shape: {x.shape}")
        
        x = self.input(x)
        # x should have shape (batch_size, #meg_channel , #timepoint)   # batch_size indicate number of event processing
        #print(f"After input layer, x.shape: {x.shape}")
        
        x = self.fn1(x)
        x = self.fn2(x)

        #print(f"After all hiden layers, x.shape: {x.shape}")
        x = self.output(x)
        # print(f"After output layer, x.shape: {x.shape}")

        # now x have dimension (batch_size, #meg_channel, 1) 
        # which indicate it made a prediction for each meg channel for each evet, which is not we want
        # we need some way compress the it to make 1 prediction for each event  
        # one way to do this is take mean of all channels
        # x = torch.mean(x, dim=1)  # Average pooling across channels
        # print(x.shape)

        return x