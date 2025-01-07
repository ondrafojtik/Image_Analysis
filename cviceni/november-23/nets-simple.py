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

#https://pytorch.org/vision/main/models/vgg.html


#vgg_11 = torchvision.models.vgg11_bn(weights="VGG11_BN_Weights.IMAGENET1K_V1")
#print(vgg_11)
#print(summary(vgg_11, (3, 224, 224), batch_size = -1))
#print("-----------------------------")

#gnet = torchvision.models.googlenet(weights="DEFAULT")
#print(gnet)
#print(summary(gnet, (3, 224, 224), batch_size = -1))
#print("-----------------------------")

#https://github.com/pytorch/vision/blob/80f41f8d32b1fcb380d5df2116063af7034ff29a/torchvision/models/resnet.py#L37-L83
rnet = torchvision.models.resnet18(weights="DEFAULT")
print(rnet)
print(summary(rnet, (3, 224, 224), batch_size = -1))
print("-----------------------------")
