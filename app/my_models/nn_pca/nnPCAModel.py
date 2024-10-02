import torch
import torch.nn as nn
import torch.functional as F

from loguru import logger

class NNPCAModel(nn.Module):
    def __init__(self, nchans: int, ntimes:int):
        super().__init__()
        self.nchans = nchans
        self.ntimes = ntimes

        # [notes!]
        # 2nd dimension of x need to = in_features of input!

        self.input = nn.Sequential(
            nn.Linear(in_features = 3, out_features =  5),
            nn.GELU()
        )
         
        self.fn1 = nn.Sequential(
            nn.Linear(in_features = 5, out_features = 2),
            nn.GELU()
        )
        
        # self.fn2 = nn.Sequential(
        #     nn.Linear(in_features = 20, out_features = 10),
        #     nn.GELU()
        # )
        
        # self.fn3 = nn.Sequential(
        #     nn.Linear(in_features = 10, out_features = 5),
        #     nn.GELU()
        # )
        # self.fn4 = nn.Sequential(
        #     nn.Linear(in_features = 5, out_features = 2),
        #     nn.GELU()
        # )
        
        
        self.output = nn.Sequential(
            nn.Linear(in_features = 2, out_features = 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # print(x.shape)
        # exit()
        
        x = self.input(x)
        
        x = self.fn1(x)
        # x = self.fn2(x)
        # x = self.fn3(x)
        # x = self.fn4(x)
 
        x = self.output(x)

        return x