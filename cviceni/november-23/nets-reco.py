# https://pytorch.org/hub/pytorch_vision_alexnet/
# https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html
# https://discuss.pytorch.org/t/why-does-alexnet-in-torch-vision-use-average-pooling/84313
# https://medium.com/analytics-vidhya/concept-of-alexnet-convolutional-neural-network-6e73b4f9ee30
# https://discuss.pytorch.org/t/alexnet-input-size-224-or-227/41272/3
# https://androidkt.com/calculating-number-parameters-pytorch-model/

import numpy as np
import cv2 as cv
from torchvision import transforms
import torchvision.models as models
import torch

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#model = models.alexnet(weights = "AlexNet_Weights.IMAGENET1K_V1")
#print(model)

#https://github.com/pytorch/vision/blob/80f41f8d32b1fcb380d5df2116063af7034ff29a/torchvision/models/resnet.py#L37-L83
#https://pytorch.org/vision/main/models/vgg.html

model = models.vgg11_bn(weights="VGG11_BN_Weights.IMAGENET1K_V1")
print("-----------------------------")
print(model)
print("-----------------------------")
#print(summary(model, (3, 224, 224), batch_size = -1))
#print("-----------------------------")

dropout = torch.nn.Dropout(p=0.5)
input = torch.ones(10) #1000
output = dropout(input)
print(output)
print(torch.mean(output))

dropout.eval()
input = torch.ones(10) #1000
output = dropout(input)
print(output)
print(torch.mean(output))

image = cv.imread("cute-shepherd-dog-posing-isolated-white-background.jpg")
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

input_tensor = preprocess(image_rgb)
#print(input_tensor.shape)

input_tensor = input_tensor.unsqueeze(0)
#print(input_tensor.shape)

model.eval()
output = model(input_tensor)
#print(output[0])
probabilities = torch.nn.functional.softmax(output[0], dim=0)
#print(probabilities)

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
