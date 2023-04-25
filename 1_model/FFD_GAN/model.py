import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
from ffd.deform import get_ffd
# G(z)

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear) or isinstance(m,nn.ConvTranspose3d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class MyDataSet(Data.Dataset):
    def __init__(self , inputs):
        super(MyDataSet, self).__init__()
        self.inputs = inputs
        # self.label  = label

    def __len__(self):
        a = self.inputs.shape[0]
        return a

    def __getitem__(self, idx):
        return self.inputs[idx]

class generator5(nn.Module):
    # initializers
    def __init__(self, noize = 12):
        super(generator5, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noize,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,256),
            nn.LeakyReLU()
        )
        self.con = nn.Sequential(
            nn.ConvTranspose3d(256,128,(4,4,4),(2,2,2),(1,1,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(128,64,(4,4,4),(2,2,1),(2,2,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(64,32,(4,4,4),(1,2,1),(1,1,2)),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(32,2,(4,4,4),(1,2,2),(1,1,2)),
            # nn.LeakyReLU()
        )
        
    # weight_init
    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def deformation(self,input):
        x = self.fc(input)
        # print('f      ',torch.mean(x))
        x = x.view(-1,256,1,1,1)
        
        x = self.con(x)
        # print('c      ',torch.mean(x))
        self.defort = x
        return x
    # forward method
    # def forward(self, input):

    def forward(self, input,base_point,base_para,shape = (18,68)):

        base_point.requires_grad_(True)
        input = input.float()
        #base_point 需要维度(2,4,8,2)第一个变化到base_point(64,3)的(0,2)坐标上,再乘上(1224,3)，最后reshape
        b  = input.size()[0]
        x = self.deformation(input)
        # print(torch.mean(x))

        x  = x.view(b,2,64)
        x  = x.transpose(1,2)        
        base_point = base_point.repeat(b,1,1)
        
        base_point[:,:,[1,2]] = base_point[:,:,[1,2]].add(x)
        base_para = base_para.repeat(b,1,1)
        x = torch.bmm(base_para,base_point).requires_grad_()

        x = x.view(b,shape[0],shape[1],3)
        
        return x


class generator6(nn.Module):
    # initializers
    def __init__(self, noize = 12):
        super(generator6, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noize,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,256),
            nn.LeakyReLU()
        )
        self.con = nn.Sequential(
            nn.ConvTranspose3d(256,128,(4,4,4),(2,2,2),(1,1,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(128,64,(4,4,4),(2,2,1),(2,2,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(64,32,(4,4,4),(1,2,1),(1,1,2)),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(32,3,(4,4,4),(1,2,2),(1,1,2)),
            nn.Tanh()
        )
        
    # weight_init
    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def deformation(self,input):
        x = self.fc(input)
        x = x.view(-1,256,1,1,1)
        x = self.con(x)
        self.defort = x
        return x
    # forward method
    # def forward(self, input):

    def forward(self, input,base_point,base_para,shape = (18,68)):
        input = input.float()
        #base_point 需要维度(2,4,8,2)第一个变化到base_point(64,3)的(0,2)坐标上,再乘上(1224,3)，最后reshape
        b  = input.size()[0]
        x = self.deformation(input)
    
        x  = x.view(b,3,64)
        x  = x.transpose(1,2)
        
        base_point = x.float()
        base_para = base_para.repeat(b,1,1).float()
        
        x = torch.bmm(base_para,base_point)

        x = x.view(b,shape[0],shape[1],3)
        
        return x

class generator7(nn.Module):
    # initializers
    def __init__(self, noize = 12):
        super(generator7, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noize,1024),
            nn.ReLU(),
            nn.Linear(1024,256),
            nn.ReLU()
        )
        self.con = nn.Sequential(
            nn.ConvTranspose2d(4,128,(4,4),(1,2),(0,2)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128,64,(5,4),(1,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,32,(5,4),(1,2),(0,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32,3,(4,3),(1,1),(0,0)),
            nn.Tanh()
        )
        
    # weight_init
    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def deformation(self,input):
        x = self.fc(input)
        x = x.view(-1,4,4,16)
        x = self.con(x)
        self.defort = x
        # x = x.permute(0,3,1,2)
        return x
    # forward method
    # def forward(self, input):

    def forward(self, input,base_point,base_para,shape = (18,68)):
        input = input.float()
        #base_point 需要维度(2,4,8,2)第一个变化到base_point(64,3)的(0,2)坐标上,再乘上(1224,3)，最后reshape
        b  = input.size()[0]

        x = self.deformation(input)
        x = x.permute(0,2,3,1)
        return x

class discriminator5(nn.Module):
    def __init__(self):
        super(discriminator5,self).__init__()
        self.con = nn.Sequential(
            nn.Conv2d(3,32,(4,4),(2,4)),
            nn.LeakyReLU(),
            nn.Conv2d(32,64,(4,4),(2,4)),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,(3,3),(2,4)),
            nn.LeakyReLU()
        )
        self.fc   = nn.Sequential(
            nn.Linear(128,512),
            nn.LeakyReLU(),
            nn.Linear(512,1)
        )
    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x ):
        x   = x.float()
        b   = x.size()[0]
        x   = x.permute(0,3,1,2)
        # print(x.shape)
        out = self.con(x)
        out = torch.reshape(out,(b,-1))
        out = self.fc(out)
        return out

class discriminator7(nn.Module):
    def __init__(self):
        super(discriminator7,self).__init__()
        self.con = nn.Sequential(
            nn.Conv2d(3,32,(4,4),(2,4)),
            nn.LeakyReLU(),
            nn.Conv2d(32,64,(4,4),(2,4)),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,(3,3),(2,4)),
            nn.LeakyReLU()
        )
        self.fc   = nn.Sequential(
            nn.Linear(128,512),
            nn.LeakyReLU(),
            nn.Linear(512,1)
        )
    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x ):
        x   = x.float()
        b   = x.size()[0]
    
        x   = x.permute(0,3,1,2)
        
        out = self.con(x)
        out = torch.reshape(out,(b,-1))
        out = self.fc(out)
        return out


if __name__ == "__main__":

    noise_dim = 12

    G = generator7(noise_dim)
    D = discriminator7()
    torch.manual_seed(1)
    G.weight_init()

    e = torch.rand([64,3,18,68])
    b = torch.rand([64,12])
    point = torch.rand([1,64,3])
    para  = torch.rand([1,1224,64])
    # print(b.shape)
    print(b.shape)
    b = G(b,point,para)
    print(b.shape)
    # exit()
    c = D(b)
    print(c.shape)

    # a = [[0,0,0],[2,2,2],[0,0,0]]
    # b = [[1,1],[1,1],[1,1]]
    # a = torch.tensor(a)
    # b = torch.tensor(b)
    # a[:,[0,2]] = a[:,[0,2]].add(b)
    # print(a)
