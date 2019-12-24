import torch 
import torchvision 

from torch import nn
from torch import optim
from torch.autograd.variable import Variable

from torchvision import transforms, datasets
from matplotlib import pyplot as plt

import os

transform  = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

mnist = datasets.MNIST(root ='./mnist_data',download = True,transform= transform)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

batch_size  = 128

EPOCH = 1000

data  = torch.utils.data.DataLoader(mnist,batch_size = batch_size,shuffle = True)

learning_rate = 0.001

clip_value = 0.01

if os.path.isdir('WGAN') == False:
  os.mkdir('WGAN')

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()

        in_feature = 784 
        hid = 256 
        out = 1

        self.hidden = nn.Sequential(nn.Linear(in_feature,hid),nn.ReLU(),nn.Dropout(0.1))
        self.out = nn.Sequential(nn.Linear(hid,out))

    def forward(self,x):
        
        x = self.hidden(x)
        x = self.out(x)
        
        return x

class Generator(nn.Module):

    def __init__(self):
        super(Generator,self).__init__()

        in_feature = 100
        hid = 256
        out = 784

        self.hidden = nn.Sequential(nn.Linear(in_feature,hid),nn.ReLU(),nn.Dropout(0.1))
        self.out = nn.Sequential(nn.Linear(hid,out),nn.Tanh())

    def forward(self,x):
        
        x = self.hidden(x) 
        x = self.out(x)

        return x
 
 
generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = optim.RMSprop(generator.parameters(), lr = learning_rate)
discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr = learning_rate*0.1)

Loss = torch.mean

def random_input(size):
    r = Variable(torch.randn(size,100)).to(device)

    return r
    

def generator_train(x):

    r = random_input(x.shape[0])

    generator_optimizer.zero_grad()
    
    fake_images = generator(r)

    fake_predict = discriminator(fake_images)

    loss = -Loss(fake_predict)

    loss.backward()

    generator_optimizer.step()

    return loss


def discriminator_train(x):

    r = random_input(x.shape[0])

    discriminator_optimizer.zero_grad()
    
    real_predict = discriminator(x)
  
   
    fake_predict = discriminator((generator(r).detach()))

    loss = -Loss(real_predict) + Loss(fake_predict)

    loss.backward()

    discriminator_optimizer.step()

    for params in discriminator.parameters():
        params.data.clamp(-clip_value , clip_value)

    return loss

def plot_images(epoch):
  images = generator(random_input(16))
  images = images.view(images.size(0), 1, 28, 28)
  figure = plt.figure()
  num_of_images = 16
  for index in range(1, num_of_images + 1):
      plt.subplot(4, 4, index)
      plt.axis('off')
      plt.imshow(images[index-1].cpu().detach().numpy().squeeze(), cmap='gray')
   
  if epoch%1 == 0 and epoch!= 0:
    plt.savefig('WGAN/Epoch_{:04d}.png'.format(epoch)) 
  plt.show()


for epoch in range(EPOCH):

    for batch , (image , label) in enumerate(data):
        
        x = image.view(image.size(0), 784).to(device)

        d_loss = discriminator_train(x)
        if batch%5==0:
            g_loss = generator_train(x)
        if batch%100==0:
            print(epoch+1,'--->',batch)
            print("Generator Loss:",g_loss,'Discriminator Loss:',d_loss)

    plot_images(epoch+1)

