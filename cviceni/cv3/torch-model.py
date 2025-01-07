import torch

#https://pytorch.org/docs/stable/generated/torch.rand.html
#https://www.transum.org/Maths/Activity/Graph/Desmos.asp
#https://vinizinho.net/projects/perceptron-viz/
#https://playground.tensorflow.org/ -------------hodne cool
#https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

features = torch.randn((1, 5))
layer = torch.nn.Linear(5, 1)
print(f"layer.weight: {layer.weight}")
print(f"layer.bias: {layer.bias}")
print(f"layer(features): {layer(features)}")


class Model_0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(16, 8),
            torch.nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layers(x) #dopredne trenovani

features = torch.randn((1, 16))
net = Model_0()
print(net)
print(net(features))

class Model_1(torch.nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(img_size*img_size, 8),
            torch.nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layers(x)

IMG_SIZE = 40
rand_img = torch.rand([1, 1, IMG_SIZE, IMG_SIZE]) # umele vytvari sum
net = Model_1(IMG_SIZE)
#f_layer = torch.nn.Flatten()
print(net(rand_img))
