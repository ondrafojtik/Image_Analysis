# https://github.com/Lyken17/pytorch-OpCounter

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import torchvision
from torchsummary import summary

from thop import profile

class NormalBlockExample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, X):
        out = self.conv(X)
        return out

class Block1x1Example(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1x1_16 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.conv_5x5_32 = nn.Conv2d(16, out_channels, kernel_size=5, padding=2)

    def forward(self, X):
        out = self.conv_1x1_16(X)
        out = self.conv_5x5_32(out)
        return out


normal_block = NormalBlockExample(192, 32)
print(summary(normal_block, (192, 28, 28), batch_size = -1))
print("-----------------------------")
my_input = torch.randn(1, 192, 28, 28)

macs, params = profile(normal_block, inputs=(my_input, ))
print("macs", macs, "params", params)

print("my_input", my_input.shape)
normal_block = NormalBlockExample(192, 32)
normal_out = normal_block(my_input)
print("normal out", normal_out.shape)
print(sum(p.numel() for p in normal_block.parameters()))
print("-----------------------------")
print("-----------------------------")
block1x1 = Block1x1Example(192, 32)
print(summary(block1x1, (192, 28, 28), batch_size = -1))
out = block1x1(my_input)
print("block1x1 out", out.shape)
print(sum(p.numel() for p in block1x1.parameters()))
print("-----------------------------")
macs, params = profile(block1x1, inputs=(my_input, ))
print("macs", macs, "params", params)

