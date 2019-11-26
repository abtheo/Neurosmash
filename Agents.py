# coding: utf-8

# ### **SOW-MKI49-2019-SEM1-V: NeurIPS**
# 
# # Project: Neurosmash
# # Group 13

import random
import socket
import struct
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn

import torch.nn.functional as F

class NeurosmashAgent(torch.nn.Module):
    def __init__(self):
        super(NeurosmashAgent, self).__init__()
        #Shape of input image
        self.state_shape = (256, 256, 3)
        #Size of action space (Nothing, Left, Right)
        self.num_actions = 3
        
        #Input channels = 3, output channels = 256
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        #256, 256, 64
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #128, 128, 64
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        #128, 128, 128
        
        #self.pool again
        #64, 64, 128

        self.linear = torch.nn.Linear(64*64*128, 128)

        self.output = torch.nn.Linear(128, self.num_actions)
        #3 output channels (Nothing=0, Left=1, Right=2)
        

    def step(self, state):
        # return 0 # no action
        # return 1 # left action
        # return 2 # right action
        # return 3 # built-in random action
        
        #TODO: Check this view transformation actually produces an image
        state = torch.tensor(state, dtype=torch.float).view(3, 256, 256).view(1,3,256,256).float()
        
        action = self.forward(state)
        
        return action
    
    def forward(self, x):
        #Convolution layer, ReLU activation
        x = F.relu(self.conv1(x))
        #MaxPooling2D
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        #Flatten pooled layer
        x = x.view(-1, 64 * 64 * 128)

        #Linear layer
        x = F.relu(self.linear(x))

        #Softmax on linear output layer
        x = F.softmax(self.output(x), dim=1)
        
        return x
