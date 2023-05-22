from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from parameter import *

channel_nums = 64
kernel_size = 5

class Residual_Block(nn.Module):
    def __init__(self, channels, kernel_size):
        super(Residual_Block,self).__init__()
        self.conv1=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=kernel_size, padding=kernel_size//2, bias=True)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=kernel_size, padding=kernel_size//2, bias=True)
        
    def forward(self,x):
        residual=x
        out=self.conv1(x)
        out=self.relu(out)
        out=self.conv2(out)
        out+=residual
        return out
    
class Net_simple(nn.Module):
    def __init__(self,CHANNEL):
        super(Net_simple, self).__init__()
        
        self.conv_HSI_input = nn.Conv2d(CHANNEL+1, channel_nums, kernel_size, padding=kernel_size//2)
        layers = []
        for i in range(10):
            layers += [Residual_Block(channel_nums, kernel_size)]
        self.layers = torch.nn.Sequential(*layers)
        self.conv_HSI_output = nn.Conv2d(channel_nums, CHANNEL, kernel_size, padding=kernel_size//2)
            
    def forward(self, coded_patch):
        
        x = self.conv_HSI_input(coded_patch)
        x = self.layers(x)
        output = self.conv_HSI_output(x)
        return output