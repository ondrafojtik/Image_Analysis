
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import cv2

# Specify the path to your main directory
data_path_ = 'train_images1'

IMG_SIZE = 40
print(data_path_)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True)])

transform2 = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5)),
     #transforms.Resize((IMG_SIZE, IMG_SIZE)),
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

transform_test = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
     transforms.Normalize((0.5), (0.5))
     ])

batch_size = 4

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                       download=True, transform=transform)


trainset  = ImageFolder(root=data_path_, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('neg_bmw', 'pos_bmw')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images


dataiter = iter(trainloader)
print(dataiter)
images, labels = next(dataiter)
print(images[0].shape)




# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.LazyLinear(84)
        self.fc3 = nn.LazyLinear(2)

    def forward(self, x):
        #x = torch.mean(x, dim=1, keepdim=True)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training
print('Started Training')

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net_bonus.pth'
torch.save(net.state_dict(), PATH)


# Testing
dataiter = iter(testloader)
images, labels = next(dataiter)

print(images[0].shape)
test___ = cv2.imread('train_images/neg_bmw/car52.jpg')
test___ = cv2.cvtColor(test___, cv2.COLOR_BGR2GRAY)
#test___ = cv2.imread('train_images/pos_bmw/augmented_image_9.jpg')
print(test___.shape)
resized = cv2.resize(test___, (40, 40), interpolation=cv2.INTER_AREA)
print(resized.shape)

PATH = './cifar_net_bonus.pth'
net = Net()
net.load_state_dict(torch.load(PATH))

tensor = transform2(resized)
tensor = tensor.unsqueeze(0)
outputs = net(tensor)


_, predicted = torch.max(outputs, 1)
string_pred = str(predicted)[8]
print(string_pred)
#print(outputs)
#print(_)
#print(predicted)


#########################
'''
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
'''
