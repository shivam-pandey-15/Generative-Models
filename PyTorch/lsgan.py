import torch
import torchvision

from torch import nn
from torch import optim
from torch.autograd.variable import Variable

from torchvision import datasets,transforms

import matplotlib.pyplot as plt
import os
import numpy as np


transform  = transforms.Compose([transforms.ToTensor(),  transforms.Normalize([0.5],[0.5])])

data = datasets.MNIST(root = './mnist_data',download = False,transform= transform)

batch_size = 128

learning_rate = 0.01

EPOCH = 1000

beta1 = 0.5
beta2 = 0.999


a , b , c = 0 , 1 , 1


data = torch.utils.data.DataLoader(data,batch_size = batch_size ,shuffle = True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if os.path.isdir('LSGAN')==False:
    os.mkdir('LSGAN')

class Generator(nn.Module):

    def __init__(self):
        
        super(Generator,self).__init__()

        # strating size 1 * 1 * 100
        self.Layer1 = nn.Sequential(
            nn.ConvTranspose2d(100 , 32 *8 , 4 , 1  , 0,bias=False),
            nn.BatchNorm2d(32*8),
            nn.ReLU() )
        # now size 4 * 4 * (32 * 8)
        self.Layer2 = nn.Sequential(
            nn.ConvTranspose2d(32 *8 , 32 * 4 ,4 , 2 , 1,bias=False),
            nn.BatchNorm2d(32*4),
            nn.ReLU())
        # now size 8 * 8 * (32 * 4)
        self.Layer3 = nn.Sequential(
            nn.ConvTranspose2d(32 *4 , 32 * 2 , 4 ,2 , 1,bias=False),
            nn.BatchNorm2d(32*2),
            nn.ReLU())
        #now size 16 * 16  * (32 *2)
        self.Layer4 = nn.Sequential(
            nn.ConvTranspose2d(32 *2 ,  1 , 4,  2 , 1,bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU())
        #final size 32 * 32  * (64 * 1)
        

    def forward(self,x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.Layer4(x)
        x = self.Layer5(x)

        return x

class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator,self).__init__()

        #starting size 32 * 32 *1
        self.Layer1 = nn.Sequential(
            nn.Conv2d(1,32,4,2,1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        #now size 16 * 16 * 32
        self.Layer2 = nn.Sequential(
            nn.Conv2d(32, 32*2,4,2,1,bias=False),
            nn.BatchNorm2d(32*2),
            nn.LeakyReLU(0.1)
        )
        #starting size 8 * 8 * (64)
        self.Layer3 = nn.Sequential(
            nn.Conv2d(32 *2 ,32 *4 ,4,2,1,bias=False),
            nn.BatchNorm2d(32*4),
            nn.LeakyReLU(0.1)
        )
        #final size 4 * 4 * (128) 
        self.Layer4 = nn.Sequential(
            nn.Linear(4 * 4 * 128 , 1)
        )
       

    def forward(self,x):
        
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = x.view(x.shape[0] , -1)
        x = self.Layer4(x)
        

        return x 



def random_input(size):
    r = Variable(torch.randn(size,100 )).to(device)
    return r

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = optim.Adam(generator.parameters(), lr = learning_rate , betas = (beta1 , beta2))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr = learning_rate*0.1 ,  betas = (beta1 , beta2))

loss = nn.MSELoss()

generator_optimizer = optim.Adam(generator.parameters(),lr = learning_rate)
discriminator_optimizer = optim.Adam(discriminator.parameters(),lr = learning_rate)

def discriminator_train(x):

    discriminator_optimizer.zero_grad()

    fake_image = generator(random_input(x.shape[0]))

    fake_loss = discriminator(fake_image.detach()) 

    real_loss = discriminator(x) 

    l = (0.50 * torch.mean((real_loss - b)**2))   + (0.50 * torch.mean((fake_loss - a)**2))
    
    l.backward()

    discriminator_optimizer.step()

    return l

def generator_train(x):

    label  = Variable(torch.ones(x.shape[0])).to(device)

    generator_optimizer.zero_grad()
    
    fake_images = discriminator(generator(random_input(x.shape[0])))

    l =  0.5 * (torch.mean(fake_images-c)**2)

    l.backward()

    generator_optimizer.step()

    return l

# def plot_images(epoch):
#   images = generator(random_input(16))
#   img = torchvision.utils.make_grid(images, nrow = 4,padding=2, normalize=True)
#   plt.imshow(np.transpose(img.detach().numpy(),(1,2,0)), animated=True)
#   plt.savefig('LSGAN/Epoch_{:4f}.png'.format(epoch))
def plot_images(epoch):
  images = generator(random_input(16))
  images = images.view(images.size(0), 1, 28, 28)
  figure = plt.figure()
  num_of_images = 16
  for index in range(1, num_of_images + 1):
      plt.subplot(4, 4, index)
      plt.axis('off')
      x = images[index-1].cpu().detach().numpy().squeeze()
      print(x.shape)
      plt.imshow(x, cmap='gray')
  if epoch%1 == 0 and epoch!= 0:
    plt.savefig('GAN/Epoch_{:04d}.png'.format(epoch)) 
  plt.show()


for epoch in range(EPOCH):

    for batch , (image , label) in enumerate(data):
        x = image.view(image.shape[0],-1).to(device)

        d_loss = discriminator_train(x)
        g_loss = generator_train(x)
        if batch%100==0:
            print(epoch+1,'--->',batch)
            print("Generator Loss:",g_loss,'Discriminator Loss:',d_loss)

    plot_images(epoch+1)



    
