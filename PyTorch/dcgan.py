import torch
import torchvision

from torch import nn
from torch import optim 
from torch.autograd.variable import Variable

from torchvision import datasets, transforms

from matplotlib import pyplot as plt 
import os
import numpy as np

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])


data = datasets.ImageFolder(root='./data/celeba' , transform = transform)

batch_size = 64

learning_rate = 0.0001

beta1 = 0.5
beta2 = 0.999

EPOCH = 100

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data = torch.utils.data.DataLoader(data,batch_size = batch_size  , shuffle = True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!= -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm')!= -1:
        nn.init.normal_(m.weight.data,1.0,0.2)
        nn.init.constant_(m.bias.data,0)

if os.path.isdir('DCGAN')==False:
    os.mkdir('DCGAN')

class Generator(nn.Module):

    def __init__(self):
        
        super(Generator,self).__init__()

        # strating size 1 * 1 * 100
        self.Layer1 = nn.Sequential(
            nn.ConvTranspose2d(100 , 64 *8 , 4 , 1  , 0,bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU() )
        # now size 4 * 4 * (64 * 8)
        self.Layer2 = nn.Sequential(
            nn.ConvTranspose2d(64 *8 , 64 * 4 ,4 , 2 , 1,bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU())
        # now size 8 * 8 * (64 * 4)
        self.Layer3 = nn.Sequential(
            nn.ConvTranspose2d(64 *4 , 64 * 2 , 4 ,2 , 1,bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU())
        #now size 16 * 16  * (64 *2)
        self.Layer4 = nn.Sequential(
            nn.ConvTranspose2d(64 *2 , 64 * 1 , 4,  2 , 1,bias=False),
            nn.BatchNorm2d(64*1),
            nn.ReLU())
        #now size 32 * 32  * (64 * 1)
        self.Layer5 = nn.Sequential(
            nn.ConvTranspose2d(64 *1 , 3 , 4 , 2 , 1),
            nn.BatchNorm2d(3),
            nn.Tanh())
        #final size 64 * 64 * 3

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

        #starting size 64 * 64 *3
        self.Layer1 = nn.Sequential(
            nn.Conv2d(3,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        #now size 32 * 32 * 64
        self.Layer2 = nn.Sequential(
            nn.Conv2d(64, 64*2,4,2,1,bias=False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.1)
        )
        #starting size 16 * 16 * (64*2)
        self.Layer3 = nn.Sequential(
            nn.Conv2d(64 *2 ,64 *4 ,4,2,1,bias=False),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.1)
        )
        #now size 8 * 8 * (64*4) 
        self.Layer4 = nn.Sequential(
            nn.Conv2d(64 *4,64 *8,4,2,1,bias=False),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.1)
        )
        #now size 4 * 4 * (64 * 8)
        self.Layer5 = nn.Sequential(
            nn.Conv2d(64 * 8 , 1, 4 , 1 , 0 ,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.Layer4(x)
        x = self.Layer5(x)

        return x


generator = Generator().to(device)
discriminator = Discriminator().to(device)
            
generator.apply(weights_init)
discriminator.apply(weights_init)

generator_optimizer = optim.Adam(generator.parameters(), lr = learning_rate , betas=  (beta1,beta2))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr = learning_rate , betas=  (beta1,beta2))

Loss = nn.BCELoss()
def random_input(size):
    r = Variable(torch.randn(size,100,1,1)).to(device)
    return r

def generator_train(x):

    r = random_input(x.shape[0])
    
    fake_label_ones = Variable(torch.ones(x.shape[0])).to(device)

    generator_optimizer.zero_grad()
    
    fake_images = generator(r)

    fake_predict = discriminator(fake_images)

    loss = Loss(fake_predict,fake_label_ones)

    loss.backward()

    generator_optimizer.step()

    return loss


def discriminator_train(x):

    r = random_input(x.shape[0])

    discriminator_optimizer.zero_grad()
    
    real_label = Variable(torch.ones(x.shape[0])).to(device)

    real_predict = discriminator(x)

    real_loss = Loss(real_predict , real_label)

    real_loss.backward()

    fake_label = Variable(torch.zeros(x.shape[0])).to(device)

    fake_predict = discriminator(generator(r).detach())

    fake_loss = Loss(fake_predict,fake_label)

    fake_loss.backward()

    loss = real_loss + fake_loss 

    discriminator_optimizer.step()

    return loss


def plot_images(epoch):
  images = generator(random_input(16))
  img = torchvision.utils.make_grid(images, nrow = 4,padding=2, normalize=True)
  plt.imshow(np.transpose(img.detach().numpy(),(1,2,0)), animated=True)
  plt.savefig('DCGAN/Epoch_{:4f}.png'.format(epoch)) 


for epoch in range(EPOCH):

    for batch , (image , label) in enumerate(data):
        print(epoch+1,'--->',batch)
        x = image.to(device)
        d_loss = discriminator_train(x)
        g_loss = generator_train(x)
        
        if batch%100==0:
            print(epoch+1,'--->',batch)
            print("Generator Loss:",g_loss,'Discriminator Loss:',d_loss)
            
    plot_images(epoch+1)