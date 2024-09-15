import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from loguru import logger

class SimpleTorchCNNModel(nn.Module):
    def __init__(self, nchans, ntimes):
        """
        notes: when i do model.to(device)
        all layers created in init of model will be automatically in the same device as the model
        """
        super(SimpleTorchCNNModel, self).__init__()
        self.nchans = nchans
        self.ntimes = ntimes

        time_window_size = 5
        channel_window_size = nchans
        layer_1_output = 32
        layer_2_output = 64

        # [notes] nn.Conv2d(input_size, output_size, kernel_size)
        self.conv1 = nn.Conv2d(1, layer_1_output, kernel_size=(1, time_window_size))
        # [notes]
        # e.g. if conv1 output =32, time window size=5, 
        #    after conv1 -> torch.Size([32, 32, 208, 37])  =   32, 32, #chan, 37 --> 41, each sliding window 5 pts so 37 window

        self.conv2 = nn.Conv2d(layer_1_output, layer_2_output, kernel_size=(channel_window_size, 1))
        # [notes]
        # e.g. if conv1 output =32, time window size=5, chan window size = # chan
        #   after conv2: torch.Size([32, 64, 1, 37])
        #   output of conv2 = (output_conv1, output_conv2, wt_left_in_chan_dim, wt_left_in_time_dim )

        
        self.flatten = nn.Flatten()
        # [notes]
        # after flatten:  torch.Size([output_conv1, multiplie_of_other_3_dim])

        # Placeholder for dynamic calculation of fc1 input size
        lin_in = (ntimes-time_window_size+1) * (nchans-channel_window_size+1) * layer_2_output
        self.fc1 = nn.Linear(lin_in, 128)         # first arg need to =output_conv2 *  wt_left_in_chan_dim * wt_left_in_time_dim
        self.fc2 = nn.Linear(128, 1)   # 128 is up to my design; Binary classification last output must be 1

    def forward(self, x):
        """
        notes: forward is automatically called on outputs = model(inputs)
        Forward pass with dynamic computation of fully connected layer input size.
        """

        x = torch.relu(self.conv1(x))
        #print(f"After conv1: {x.shape}")  # Inspect dimensions after conv1

        x = torch.relu(self.conv2(x))
        #print(f"After conv2: {x.shape}")  # Inspect dimensions after conv2
        
        x = self.flatten(x)
        #print(f"After flatten: {x.shape}")  # Inspect dimensions after flattening
        
        
        # Pass through fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

