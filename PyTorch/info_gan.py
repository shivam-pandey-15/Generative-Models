import torch
import torchvision

from torch import nn
from torch import optim
from torch.autograd.variable import Variable

from torchvision import datasets
from torchvision import transforms

from matplotlib import pyplot as plt
import os
import itertools
import numpy as np

transform = transforms.Compose([transforms.ToTensor() ,transforms.Normalize([0.5],[0.5])])

data  = datasets.MNIST(root='./mnist_data',download =True , transform = transform)

c =10

z =62

EPOCH = 1000

batch_size = 128

learning_rate = 0.01

data = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle = True)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if os.path.isdir('INFOGAN')==False:
    os.mkdir('INFOGAN')


class Generator(nn.Module):

    def __init__(self):
        super(Generator,self).__init__()


        self.FC1 = nn.Sequential(nn.Linear(74,1024),nn.ReLU())
        self.bn1 = nn.BatchNorm1d(1024)

        self.FC2 = nn.Sequential(nn.Linear(1024,7*7*128),nn.ReLU())
        self.bn2 = nn.BatchNorm1d(7 * 7 * 128)

        self.Conv1 = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1),nn.ReLU())
        self.bn3  = nn.BatchNorm2d(64)

        self.Conv2 = nn.Sequential(nn.ConvTranspose2d(64,1,4,2,1),nn.Tanh())


    def forward(self,x):
        
        x = self.FC1(x)        
        x = self.bn1(x)
        x = self.FC2(x)
        x = self.bn2(x)
        x  = x.view(-1,128,7,7)
        x = self.Conv1(x)
        x = self.bn3(x)
        x= self.Conv2(x)
        
        return x
class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator,self).__init__()


        self.Conv1 = nn.Sequential(nn.Conv2d(1,64,4,2,1),nn.LeakyReLU(0.1))
        self.Conv2 =  nn.Sequential(nn.Conv2d(64,128,4,2,1),nn.LeakyReLU(0.1),nn.BatchNorm2d(128))
        self.fc1 =  nn.Sequential(nn.Linear(7*7*128,1024),nn.LeakyReLU(0.1),nn.BatchNorm1d(1024))
        self.fc_discriminator =  nn.Sequential(nn.Linear(1024,1),nn.Sigmoid())

        self.fc_type = nn.Sequential(nn.Linear(1024,2),nn.BatchNorm1d(2),nn.LeakyReLU(0.1))
        self.fc_class = nn.Sequential(nn.Linear(1024,10),nn.BatchNorm1d(10),nn.LeakyReLU(0.1))
        


    def forward(self,x):
        
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = x.view(-1,7*7*128)
        x = self.fc1(x)
        x_d =  self.fc_discriminator(x)
        x_type = self.fc_type(x)
        x_dim = self.fc_class(x)
        
        return x_d,x_type,x_dim


generator = Generator().to(device)
discriminator = Discriminator().to(device)



generator_optimizer = optim.Adam(generator.parameters() , lr = learning_rate)
discriminator_optimizer = optim.Adam(discriminator.parameters() , lr = learning_rate)
q_optimizer = optim.Adam(itertools.chain(discriminator.parameters(),generator.parameters()) , lr =learning_rate)



def random_input(size,var):

    r  = Variable(torch.rand(size,var)).to(device)
    return r

def categorical_input(size,var):
    y = np.random.randint(0,var,size)
    y_cat = np.zeros((y.shape[0], var))
    y_cat[range(y.shape[0]), y] = 1.0
    return Variable(torch.tensor(y_cat,dtype = torch.float32)).to(device)

def generator_train(x):

    r = random_input(x.shape[0],z)
    label = categorical_input(x.shape[0],c)
    dim = Variable(torch.tensor(np.random.uniform(-1, 1, (x.shape[0], 2)),dtype = torch.float32)).to(device)

    r = torch.cat([r,label,dim],dim=1)
    
    fake_label_ones = Variable(torch.ones(x.shape[0])).to(device)

    generator_optimizer.zero_grad()
    
    fake_images = generator(r)

    fake_predict,_,_ = discriminator(fake_images)

    Loss = torch.nn.BCELoss()

    loss = Loss(fake_predict,fake_label_ones)

    loss.backward()

    generator_optimizer.step()

    return loss


def discriminator_train(x):

    r = random_input(x.shape[0],z)
    label = categorical_input(x.shape[0],c)
    dim = Variable(torch.tensor(np.random.uniform(-1, 1, (x.shape[0], 2)),dtype = torch.float32)).to(device)
   
    r = torch.cat([r,label,dim],dim=1)

    discriminator_optimizer.zero_grad()
    

    Loss = torch.nn.BCELoss()

    real_label = Variable(torch.ones(x.shape[0])).to(device)

    real_predict,_,_ = discriminator(x)

    real_loss = Loss(real_predict , real_label)

    real_loss.backward()

    fake_label = Variable(torch.zeros(x.shape[0])).to(device)

    fake_predict,_,_ = discriminator(generator(r).detach())

    fake_loss = Loss(fake_predict,fake_label)

    fake_loss.backward()

    loss = (real_loss + fake_loss)/2 

    discriminator_optimizer.step()

    return loss

def q_train(x):

    r = random_input(x.shape[0],z)
    label = categorical_input(x.shape[0],c)
    dim = Variable(torch.tensor(np.random.uniform(-1, 1, (x.shape[0], 2)),dtype = torch.float32)).to(device)

    r = torch.cat([r,label,dim],dim=1)

    lambda_label = 1

    lambda_dimmension = 0.2

    q_optimizer.zero_grad()

    fake_images = generator(r)

    _ , pred_dim , pred_label  = discriminator(fake_images)

    act_label = label = Variable(torch.LongTensor(np.random.randint(0, 10, x.shape[0])),requires_grad=False).to(device) 

    Loss_label = torch.nn.CrossEntropyLoss()
    Loss_dim = torch.nn.MSELoss()
    
    
    loss = lambda_label * Loss_label(pred_label, act_label) + lambda_dimmension * Loss_dim(pred_dim,dim  )
    

    loss.backward()

    q_optimizer.step()
    

def plot_images(epoch):
  r = random_input(16,z)
  label = categorical_input(16,c)
  dim = Variable(torch.tensor(np.random.uniform(-1, 1, (16, 2)),dtype=torch.float32)).to(device)
  r = torch.cat([r,label,dim],dim=1)
  images = generator(r)
  images = images.view(images.size(0), 1, 28, 28)
  figure = plt.figure()
  num_of_images = 16
  print(label)
  for index in range(1, num_of_images + 1):
      plt.subplot(4, 4, index)
      plt.axis('off')
      x = images[index-1].cpu().detach().numpy().squeeze()
      plt.imshow(x, cmap='gray')
  if epoch%1 == 0 and epoch!= 0:
    plt.savefig('INFOGAN/Epoch_{:04d}.png'.format(epoch)) 
  plt.show()

#-----------------------Training--------------------#

for epoch in range(EPOCH):

    for batch , (image , label) in enumerate(data):
        
        x = image.to(device)

        d_loss = discriminator_train(x)
        g_loss = generator_train(x)
        q_loss = q_train(x)
        if batch%100==0:
            print(epoch+1,'--->',batch)
            print("Generator Loss:",g_loss,'Discriminator Loss:',d_loss)

    plot_images(epoch+1)

