import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

IMG_SIZE = 64

label_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
     ])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)

import matplotlib.pyplot as plt
import numpy as np

figure = plt.figure(figsize=(8, 8))
for i in range(1, 26):
    img, label = trainset[np.random.randint(0, len(trainset))]
    figure.add_subplot(5, 5, i)
    print(img.size())
    img_t = np.transpose(img, (1, 2, 0))
    print(type(img))
    #img_t = torch.squeeze(img, 1)
    print(img_t.size())
    plt.title(label_map[label])
    plt.axis("off")
    plt.imshow(img_t, cmap="gray")
    print(i)
plt.show()
