import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import torchvision
from torchsummary import summary
from thop import profile
from thop import clever_format

class ConventionalBlockExample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=0, stride=1)

    def forward(self, X):
        out = self.conv(X)
        return out

class DepthwiseBlockExample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.kernels_per_layer = 1
        self.depthwise = nn.Conv2d(in_channels, in_channels * self.kernels_per_layer, kernel_size=5, padding=0, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels*self.kernels_per_layer, out_channels, kernel_size=1)

    def forward(self, X):
        #print("X.shape", X.shape)
        out = self.depthwise(X)
        #print("depthwise.shape", out.shape)
        out = self.pointwise(out)
        return out

print("------------------- CASE - 0")
batch = 1
in_channels = 1
in_size = 12
out_channels = 1

my_input = torch.randn(batch, in_channels, in_size, in_size)
conv_block = ConventionalBlockExample(in_channels, out_channels)
conv_block_out = conv_block(my_input)

print("conv_block_out.shape", conv_block_out.shape)
macs, params = profile(conv_block, inputs=(my_input, ))
#macs, params = clever_format([macs, params], "%.3f")
print("macs", macs, "params", params)


print("------------------- CASE - 1")
batch = 1
in_channels = 3
in_size = 12
out_channels = 1

my_input = torch.randn(batch, in_channels, in_size, in_size)
conv_block = ConventionalBlockExample(in_channels, out_channels)
conv_block_out = conv_block(my_input)

from thop import clever_format

print("conv_block_out.shape", conv_block_out.shape)
macs, params = profile(conv_block, inputs=(my_input, ))
#macs, params = clever_format([macs, params], "%.3f")
print("macs", macs, "params", params)


print("------------------- CASE - 2")
batch = 1
in_channels = 3
in_size = 12
out_channels = 128

my_input = torch.randn(batch, in_channels, in_size, in_size)
conv_block = ConventionalBlockExample(in_channels, out_channels)
conv_block_out = conv_block(my_input)

from thop import clever_format

print("conv_block_out.shape", conv_block_out.shape)
macs, params = profile(conv_block, inputs=(my_input, ))
#macs, params = clever_format([macs, params], "%.3f")
print("macs", macs, "params", params)


print("------------------- CASE - 3")
batch = 1
in_channels = 3
in_size = 12
out_channels = 128

my_input = torch.randn(batch, in_channels, in_size, in_size)
conv_block = DepthwiseBlockExample(in_channels, out_channels)
conv_block_out = conv_block(my_input)

print("conv_block_out.shape", conv_block_out.shape)
macs, params = profile(conv_block, inputs=(my_input, ))
#macs, params = clever_format([macs, params], "%.3f")
print("macs", macs, "params", params)
