import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

IMG_SIZE = 40

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


print(f"trainset: {trainset}")
print(f"testset: {testset}")
print(f"trainset[0]: {trainset[0]}")

# last element in list - label_map_id
#img, label = trainset[0]
#print(img.shape)

import matplotlib.pyplot as plt
import numpy as np

# flip arrays (refering to img.shape)
#img_tr = torch.permute(1, 2, 0) # na jake pozici ma byt jaky kanal
#plt.imshow(img_tr)
#plt.show()

###############################




###############################

#x = torch.rand(size=(1,5)) # array
#print(f"x: {x}")
#w = torch.rand(size=(1,5)) # weight
#print(f"w: {w}")
#b = torch.rand(size=(1,1)) # bias
#print(f"b: {b}")
#out = torch.mm(x,w.T) + b
#print(f"out: {out}") # shapes cannot be multiplied (1x5 & 1x5) -> transpone
#out_sigmoid = torch.sigmoid(out)
#print(f"out_sigmoid: {out_sigmoid}")
#
#
#features = torch.randn((1, 10))
## dela to stejne jako predchozi vec s tou tranponovanou matici
#layer = torch.nn.Linear(10, 1)          # 1 vrstva size
#print(f"layer.weight: {layer.weight}")
#print(f"layer.bias: {layer.bias}")
#print(f"layer(features): {layer(features)}")

'''
vice vrstev by vypadalo tahle
layer = torch.nn.Linear(5, 1), torch.nn.Linear(5, 1), torch.nn.Linear(5, 1), torch.nn.Linear(5, 1), ..
musi na sebe ale sedet ty cisla jako u matice -> viz torch-model.py

torch.nn.Flatten() -> obraz je 2D, potrebujeme jednorozmerne data namapovane na ten obraz

'''

'''

IMG_SIZE = 40
rand_img = torch.rand([1, 1, IMG_SIZE, IMG_SIZE]) # umele vytvari sum
net = Model_1(IMG_SIZE)
f_layer = torch.nn.Flatten()
print(f_layer(rand_img).shape)
print(net(rand_img))

'''


###############################

# 10 -> mame 10 druhu -> 10 labelu

class Model_1(torch.nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(img_size*img_size, 10),
            torch.nn.Linear(10, 10)
        )

    def forward(self, x):
        return self.layers(x)


import torch
import torch.optim as optim


BATH_SIZE = 16 # batch size?

rand_img = torch.rand([1, 1, IMG_SIZE, IMG_SIZE]) # umele vytvari sum
net = Model_1(IMG_SIZE)
f_layer = torch.nn.Flatten()
print("----------------------------------")
print(f_layer(rand_img).shape)
print("----------------------------------")
print(net(rand_img))
print("----------------------------------")


# tohle je to co jsme delal predtim na hodne radku, vybira proste data
# mozna nefunguje na windows (kvuli toho workers)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATH_SIZE,
                                          shuffle=True, num_workers=0)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

'''
for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        #print(len(labels))
        #print(len(inputs))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print(len(outputs))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = './model1.pth'
torch.save(net.state_dict(), PATH)
'''

PATH = './model1.pth'
net = Model_1(IMG_SIZE)
net.load_state_dict(torch.load(PATH))


figure = plt.figure(figsize=(12, 12))
for i in range(1, 26):
    img, label = testset[np.random.randint(0, len(testset))]
    
    outputs = net(img)
    _, predicted = torch.max(outputs, 1)
    string_pred = str(predicted)[8]
    print(string_pred)

    figure.add_subplot(5, 5, i)
    print(img.size())
    img_t = np.transpose(img, (1, 2, 0))
    print(type(img))
    #img_t = torch.squeeze(img, 1)
    print(img_t.size())
    string_title = label_map[label] + " - " + label_map[int(string_pred)]
    plt.title(string_title)
    plt.axis("off")
    plt.imshow(img_t, cmap="gray")
    print(i)
plt.show()







plt.show()
# ta sit se da ulozit do souboru po natrenovani
# vyhodnoceni udelat doma
