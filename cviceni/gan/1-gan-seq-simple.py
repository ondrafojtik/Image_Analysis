#https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE = 64
BATCH_SIZE = 64

transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
     transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
     transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])
     ])

train_dataset = torchvision.datasets.ImageFolder(root='train_images', transform=transform)


import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

   
class Discriminator_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # We don't use Conv layers here but we vectorize our inputs
            nn.Flatten(),
            nn.Linear(IMG_SIZE*IMG_SIZE, 2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output

class Generator_0(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, IMG_SIZE*IMG_SIZE),
            #nn.Sigmoid(),
            nn.Tanh(), # We use the Tanh() activation fucntion so that our outputs lie between -1 and 1
        )

    def forward(self, x):
        output = self.model(x)
        #output = output.view(x.size(0), 1, 28, 28)
        return output

g_net = Generator_0().to(device)
d_net = Discriminator_0().to(device)

print(g_net)
print(d_net)

noise_img = torch.randn(32, 100).to(device)
fake = g_net(noise_img)
#plt.imshow(fake[0,:].detach().squeeze().view(28,28))
#plt.show()
print(fake.shape)


import torch.optim as optim
lossfun = torch.nn.BCELoss()
d_optimizer = torch.optim.Adam(d_net.parameters(), lr=.0002)
g_optimizer = torch.optim.Adam(g_net.parameters(), lr=.0002)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                          shuffle=True, drop_last=True)
for epoch in range(2000):  # loop over the dataset multiple times
    print(epoch)   
    for i, data in enumerate(trainloader, 0):

        real_img, labels = data[0].to(device), data[1].to(device)
        fake_img = g_net(torch.randn(BATCH_SIZE, 100).to(device))
        
        real_labels = torch.ones(BATCH_SIZE,1).to(device)
        fake_labels = torch.zeros(BATCH_SIZE,1).to(device)        
        
        ### ---------------- Train the discriminator ---------------- ###

        # forward pass and loss for REAL pictures
        pred_real   = d_net(real_img)              # REAL images into discriminator
        d_loss_real = lossfun(pred_real,real_labels) # all labels are 1

        # forward pass and loss for FAKE pictures
        pred_fake   = d_net(fake_img)              # FAKE images into discriminator
        d_loss_fake = lossfun(pred_fake,fake_labels) # all labels are 0

        # collect loss (using combined losses)
        d_loss = d_loss_real + d_loss_fake

        # backprop
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        ### ---------------- Train the generator ---------------- ###

        # create fake images and compute loss
        fake_images = g_net( torch.randn(BATCH_SIZE, 100).to(device) )
        pred_fake   = d_net(fake_images)
        
        # compute and collect loss and accuracy
        g_loss = lossfun(pred_fake,real_labels)

        # backprop
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    #img = fake_images[0].view(IMG_SIZE,IMG_SIZE)
    #torchvision.utils.save_image(img,f"1-gan-out/{epoch}_fake_images.jpg", normalize=True)
    with torch.no_grad():       
        fake_progress = g_net(noise_img) #g_net(torch.randn(BATCH_SIZE, nz, 1, 1).to(device))#
        print("fake_progress", fake_progress.shape)
        img = fake_progress.view(32, 1, IMG_SIZE,IMG_SIZE)
        grid_img = torchvision.utils.make_grid(img, nrow=8)
        torchvision.utils.save_image(grid_img,f"out/{epoch}e_g_images.jpg", normalize=True)        

# generate the images from the generator network
g_net.eval()
fake_data = g_net(torch.randn(26,100).to(device))

figure = plt.figure(figsize=(8, 8))
for i in range(1, 26):
    print(f"tensor shape: {fake_data[i].size()}")
    figure.add_subplot(5, 5, i)
    img = fake_data[i].cpu().detach().view(IMG_SIZE,IMG_SIZE)
    print(f"img shape: {img.size()}")
    #print(type(img))
    #img_t = torch.squeeze(img, 1)
    #print("img_t.size()", img_t.size())
    #plt.title(label_map[label])
    #plt.axis("off")
    plt.imshow(img, cmap='gray')    
    print(i)
plt.show()

fig,axs = plt.subplots(3,4,figsize=(8,6))
for i,ax in enumerate(axs.flatten()):
  ax.imshow(fake_data[i,:,].cpu().detach().view(IMG_SIZE,IMG_SIZE),cmap='gray')
  ax.axis('off')

plt.show()

