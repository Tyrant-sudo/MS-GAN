import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as Data
import time
import torch.optim as optim
import torch.autograd as autograd
import random

import matplotlib.pyplot as plt
import itertools

from method import PCA_vari
from ffd import deform 
from method import draw_geom,calc_gradient_penalty,draw_block,easy_save


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
    
def getlowDataMat(DataMat, K_eigenVector):
    return torch.bmm(DataMat , K_eigenVector)

def Reconstruction(lowDataMat, K_eigenVector, meanVal):

    K_eigenVector = K_eigenVector.transpose(1,2)
    reconDataMat = torch.bmm(lowDataMat , K_eigenVector).add(meanVal)
    return reconDataMat

class generate_process():
    def __init__(self,batch_size = 64,noise_size = 32,mesh2num = 32,device = 'cuda',low = 11,upper = 44) -> None:

        self.dim   = (18,68,3)
        self.ffd_dim   = (3,7,1)
        self.batchsize  = batch_size
        self.noise_size = noise_size
        self.mesh2num   = mesh2num

        self.device     = device
        self.low        = low
        self.upper      = upper
        self.vari_Vnum  = upper - low

    def get_input(self,base_mesh,mesh2num_model):
        
        ffd_dim = self.ffd_dim
        self.point_num = (ffd_dim[0]+1)*(ffd_dim[1]+1)*(ffd_dim[2]+1)
        self.base  = base_mesh.float() #(18,68,3)
        device = self.device
        base = self.base.to(device)

        extract_model = mesh2num_model.to(device)
        self.mesh_charater = extract_model(base).repeat(self.batchsize,1)
        
        self.input_dim = self.noise_size + self.mesh2num

    def get_ffd(self):

        ffd_dim = self.ffd_dim
        base_flat = self.base.cpu().numpy().reshape(-1,3)
        
        self.base_para,self.base_point = deform.get_ffd(base_flat,ffd_dim)

    def get_PCA(self):
        
        upper = self.upper
        low   = self.low
        self.vari_Vnum = upper - low
        V, meanVal,dataMat = PCA_vari(self.base_para,upper)
        
        self.variable_V = V[:,low:]
        self.stable_V   = V[:,:low]

        self.V = V
        self.meanVal   = meanVal
        self.para_mat = dataMat

    def get_variation(self,generator,po):
        # generator生成(point_num,2)的向量vari_point和(piont_num,V_num)的向量vari_para,输入为(64)
        device = self.device
        # print(self.noise_size)
        self.noise      = np.random.randn(self.batchsize,self.noise_size) 
        
        noise_mean = np.mean(self.noise[:,po])
        
        tmp  = np.linspace(0.5*noise_mean,1.5*noise_mean,self.batchsize)
        self.noise[:,po] = tmp
        self.noise = torch.from_numpy(self.noise).float()

        self.noise = self.noise.to(device)


        self.generator_input = torch.cat([self.mesh_charater,self.noise],1)
        
        self.vari1,self.vari2 = generator(self.generator_input)
        # self.vari2 = self.vari2[:,self.low:]

        self.vari1.to(self.device)
        self.vari2.to(self.device)

        # self.vari1 = torch.randn(self.batchsize,self.point_num,2).to(self.device) * 0.01
        # self.vari2 = torch.randn(self.batchsize,self.point_num,self.vari_Vnum).to(self.device) *0.01

    def easy_get_variation(self,generator):
        # generator生成(point_num,2)的向量vari_point和(piont_num,V_num)的向量vari_para,输入为(64)
        device = self.device
        # print(self.noise_size)
        self.noise      = torch.randn(self.batchsize,self.noise_size) 
        self.noise = self.noise.to(device)
        self.generator_input = self.noise
        self.vari1,self.vari2 = generator(self.generator_input)
        self.vari1.to(self.device)
        self.vari2.to(self.device)

    def generate_mesh_old(self):
        device = self.device
        ffd_dim    = self.ffd_dim

        base_point  = self.base_point
        batch_point = torch.from_numpy(base_point[None,:,:]).repeat(self.batchsize,1,1).to(device)
        batch_point[:,:,[1,2]] = batch_point[:,:,[1,2]].add(self.vari1)

        batch_vari   = torch.from_numpy(self.variable_V[None,:,:]).repeat(self.batchsize,1,1).to(device)
        batch_vari   =  batch_vari.add(self.vari2)
        
        batch_stable = torch.from_numpy(self.stable_V[None,:,:]).repeat(self.batchsize,1,1).to(device)
        batch_V      = torch.cat([batch_stable,batch_vari],2)

        batch_meanVal = torch.from_numpy(self.meanVal[None,:,:]).repeat(self.batchsize,1,1).to(device)
        batch_paramat = torch.from_numpy(self.para_mat[None,:,:]).repeat(self.batchsize,1,1).to(device)

        batch_lowmat  = getlowDataMat(batch_paramat,batch_V)
        
        batch_para   = Reconstruction(batch_lowmat,batch_V,batch_meanVal)
        
        batch_para,batch_point = batch_para.double(),batch_point.double()
        self.batch_mesh   = torch.bmm(batch_para,batch_point)
        
        return self.batch_mesh.cpu().detach().numpy().reshape(self.batchsize,18,68,3)
    
    def generate_mesh(self):
        device = self.device

        base_point  = self.base_point
        batch_point = torch.from_numpy(base_point[None,:,:]).repeat(self.batchsize,1,1).to(device)
        batch_point[:,:,[1,2]] = batch_point[:,:,[1,2]].add(self.vari1)

        batch_V_old   = torch.from_numpy(self.V[None,:,:]).repeat(self.batchsize,1,1).to(device)
        batch_V       = batch_V_old * (1 + self.vari2)

        batch_meanVal = torch.from_numpy(self.meanVal[None,:,:]).repeat(self.batchsize,1,1).to(device)
        batch_paramat = torch.from_numpy(self.para_mat[None,:,:]).repeat(self.batchsize,1,1).to(device)

        batch_lowmat  = getlowDataMat(batch_paramat,batch_V)
        
        batch_para   = Reconstruction(batch_lowmat,batch_V,batch_meanVal)
        
        batch_para,batch_point = batch_para.double(),batch_point.double()
        self.batch_mesh   = torch.bmm(batch_para,batch_point)
        
        return self.batch_mesh.cpu().detach().numpy().reshape(self.batchsize,18,68,3)

class process2():
    # 用于网络的训练、记录与验证
    def __init__(self,process1:generate_process,m2n,G,D,samples = 100,epochs = 3000,lrG = 0.0001,lrD = 0.0004,\
                 process_path = 'result/process/',verify_path = 'result/last_result/') -> None:
        
        self.process1 = process1
        
        self.D_losses = []
        self.G_losses = []
        self.D_real   = []
        self.D_fake   = []
        self.R3s      = []
        
        self.m2n = m2n
        self.G = G
        self.D = D
        
        self.G_optimizer = optim.RMSprop(itertools.chain(self.m2n.parameters(),self.G.parameters()),lr = lrG)
        self.D_optimizer = optim.RMSprop(self.D.parameters(),lr = lrD)
        self.path1 = process_path
        self.path2 = verify_path
        
        self.epochs   = epochs

        self.samples = samples
        c = random.sample(range(1,self.epochs),self.samples)
        self.sequence = c

    
    def train_one_epoch(self,epoch,train_loader,opG_frequency = 2):
        self.epoch_start_time = time.time()
        cur_basemesh = torch.from_numpy(train_loader.dataset[self.sequence[epoch%self.samples]])

        self.m2n.train()
        self.G.train()
        self.D.train()
        if (epoch+1) == int(self.epochs/5):
            self.G_optimizer.param_groups[0]['lr'] /= 10
            self.D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        if (epoch+1) == int(self.epochs/2):
            self.G_optimizer.param_groups[0]['lr'] /= 10
            self.D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        self.process1.get_input(cur_basemesh,self.m2n)
        self.process1.get_ffd()
        self.process1.get_PCA()
        self.epoch_start_time = time.time()

        for _,real_mesh in enumerate(train_loader):
            for p in self.D.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            self.D_optimizer.zero_grad() 
        
            real_mesh = real_mesh.to(self.process1.device)
            real_score   = self.D(real_mesh)

            d_loss_real  = torch.mean(real_score) 
            (-d_loss_real).backward(retain_graph=True)

            self.process1.get_variation(self.G)
            self.process1.generate_mesh()
            fake_mesh    = self.process1.batch_mesh.float()
            fake_mesh    = autograd.Variable(fake_mesh)
        
            fake_score   = self.D(fake_mesh)
            d_loss_fake  = torch.mean(fake_score)
            (d_loss_fake).backward(retain_graph=True)
            
            gradient_penalty =  calc_gradient_penalty(self.D, real_mesh.data, fake_mesh.data)
            gradient_penalty.backward(retain_graph=True)

            d_loss = - d_loss_real + d_loss_fake + gradient_penalty
            self.D_optimizer.step() 

            for p in self.D.parameters():
                p.requires_grad = False

            self.D_losses.append(d_loss.item())
            self.D_real.append(d_loss_real.item())
            self.D_fake.append(d_loss_fake.item())

            for j in range(opG_frequency):

                self.G_optimizer.zero_grad()
                self.process1.get_variation(self.G)
                fake_mesh = self.process1.generate_mesh()
                defort1  = self.process1.vari1
                defort2  = self.process1.vari2

                R3       = 1000*(torch.mean(torch.pow(defort1,2)) + torch.mean(torch.pow(defort2,2)))
                R3       = autograd.Variable(R3)
                R3.requires_grad_(True)
                R3.backward(retain_graph=True)

                fake_score = self.D(self.process1.batch_mesh)

                g_loss = -torch.mean(fake_score)
                g_loss = autograd.Variable(g_loss)
                g_loss.requires_grad_(True)
                g_loss.backward()

                self.G_optimizer.step()
                self.G_losses.append(g_loss.item())
                self.R3s.append(R3.item())

    def train_easy(self,epoch,train_loader,cur_basemesh,opG_frequency = 2):
        self.epoch_start_time = time.time()

        self.m2n.train()
        self.G.train()
        self.D.train()
        if (epoch+1) == int(self.epochs/5):
            self.G_optimizer.param_groups[0]['lr'] /= 10
            self.D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        if (epoch+1) == int(self.epochs/2):
            self.G_optimizer.param_groups[0]['lr'] /= 10
            self.D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        self.process1.get_input(cur_basemesh,self.m2n)
        self.process1.get_ffd()
        self.process1.get_PCA()
        self.epoch_start_time = time.time()

        for _,real_mesh in enumerate(train_loader):
            for p in self.D.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            self.D_optimizer.zero_grad() 
        
            real_mesh = real_mesh.to(self.process1.device)
            real_score   = self.D(real_mesh)

            d_loss_real  = torch.mean(real_score) 
            (-d_loss_real).backward(retain_graph=True)

            self.process1.easy_get_variation(self.G)
            self.process1.generate_mesh()
            fake_mesh    = self.process1.batch_mesh.float()
            fake_mesh    = autograd.Variable(fake_mesh)
        
            fake_score   = self.D(fake_mesh)
            d_loss_fake  = torch.mean(fake_score)
            (d_loss_fake).backward(retain_graph=True)
            
            gradient_penalty =  calc_gradient_penalty(self.D, real_mesh.data, fake_mesh.data)
            gradient_penalty.backward(retain_graph=True)

            d_loss = - d_loss_real + d_loss_fake + gradient_penalty
            self.D_optimizer.step() 

            for p in self.D.parameters():
                p.requires_grad = False

            self.D_losses.append(d_loss.item())
            self.D_real.append(d_loss_real.item())
            self.D_fake.append(d_loss_fake.item())

            for j in range(opG_frequency):

                self.G_optimizer.zero_grad()
                self.process1.easy_get_variation(self.G)
                fake_mesh = self.process1.generate_mesh()
                defort1  = self.process1.vari1
                defort2  = self.process1.vari2

                R3       = (torch.mean(torch.pow(defort1,2)) + torch.mean(torch.pow(defort2,2)))
                R3       = autograd.Variable(R3)
                R3.requires_grad_(True)
                R3.backward(retain_graph=True)

                fake_score = self.D(self.process1.batch_mesh)

                g_loss = -torch.mean(fake_score)
                g_loss = autograd.Variable(g_loss)
                g_loss.requires_grad_(True)
                g_loss.backward()

                self.G_optimizer.step()
                self.G_losses.append(g_loss.item())
                self.R3s.append(R3.item())

    def easy_draw(self,epoch):
        fake_mesh = self.process1.generate_mesh()
        t = np.random.randint(0,fake_mesh.shape[0])
        fake_mesh = fake_mesh[t]

        easy_save(fake_mesh,self.path1 + '/'+str(epoch).zfill(4))
        D_loss = torch.mean(torch.FloatTensor(self.D_losses))
        G_loss = torch.mean(torch.FloatTensor(self.G_losses))
        D_real = torch.mean(torch.FloatTensor(self.D_real))
        D_fake = torch.mean(torch.FloatTensor(self.D_fake))
        R3     = torch.mean(torch.FloatTensor(self.R3s))

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - self.epoch_start_time
        print('[%d/%d] - ptime: %.2f, loss_d: %.3f,loss_d_real:  %.3f,loss_d_fake:   %.3f, loss_g: %.3f , R3 : %.3f' % \
                                                                ((epoch + 1), self.epochs,per_epoch_ptime, 
                                                                D_loss,
                                                                D_real,
                                                                D_fake,
                                                                G_loss,
                                                                R3))
    def draw_save(self,epoch):
        
        fake_mesh = self.process1.generate_mesh()
        t = np.random.randint(0,fake_mesh.shape[0])
        fake_mesh = fake_mesh[t]

        draw_geom(fake_mesh,str(epoch)+'_mesh',self.path1,-20,15)
        draw_block(fake_mesh,str(epoch)+'_cR',self.path1)

        torch.save(self.G.state_dict(), self.path2 +"generator_param.pkl")
        torch.save(self.D.state_dict(), self.path2 +"discriminator_param.pkl")
        torch.save(self.m2n.state_dict(), self.path2 +"m2n_param.pkl")
        
        D_loss = torch.mean(torch.FloatTensor(self.D_losses))
        G_loss = torch.mean(torch.FloatTensor(self.G_losses))
        D_real = torch.mean(torch.FloatTensor(self.D_real))
        D_fake = torch.mean(torch.FloatTensor(self.D_fake))
        R3     = torch.mean(torch.FloatTensor(self.R3s))

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - self.epoch_start_time
        print('[%d/%d] - ptime: %.2f, loss_d: %.3f,loss_d_real:  %.3f,loss_d_fake:   %.3f, loss_g: %.3f , R3 : %.3f' % \
                                                                ((epoch + 1), self.epochs,per_epoch_ptime, 
                                                                D_loss,
                                                                D_real,
                                                                D_fake,
                                                                G_loss,
                                                                R3))
        


        np.save(self.path2+'D_losses',self.D_losses)
        np.save(self.path2+'G_losses',self.G_losses)
        np.save(self.path2+'D_real',self.D_real)
        np.save(self.path2+'D_fake',self.D_fake)

        plt.plot(self.D_losses,label= 'D_loss')
        plt.plot(self.G_losses,label= 'G_loss')
        plt.legend()
        plt.savefig(self.path2 + 'training_process.png')
        plt.close()

    def sample(self,base_mesh,po= 0,test_path = 'result/test_pic',draw = True):

        process1 = self.process1

        process1.get_input(base_mesh,self.m2n)
        process1.get_ffd()
        process1.get_PCA()
        
        process1.get_variation(self.G,po)

        fake_mesh = process1.generate_mesh()
        np.save(test_path+'/fake_mesh',fake_mesh)
        
        if draw:
            draw_block(base_mesh,'base',test_path)
            for _,i in enumerate(fake_mesh):
                draw_block(i,str(_),test_path)
                draw_geom(i,str(_)+'_mesh',test_path,-20,15)
        return fake_mesh
    
    def easy_sample(self,base_mesh,test_path = 'result/test_pic',draw = True):

        process1 = self.process1

        process1.get_input(base_mesh,self.m2n)
        process1.get_ffd()
        process1.get_PCA()
        
        process1.easy_get_variation(self.G)

        fake_mesh = process1.generate_mesh()
        np.save(test_path+'/fake_mesh',fake_mesh)
        
        if draw:
            for _,i in enumerate(fake_mesh):
                easy_save(i,test_path+'/'+str(_))

        return fake_mesh

class ConvNet(nn.Module):
    # 输入 
    def __init__(self,out_size):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 9 * 34, out_size)

    def forward(self, x):
        x = x.view(1,3,18,68)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 9 * 34)
        x = self.fc1(x)
        return x

class DeepConvNet(nn.Module):
    def __init__(self,input_dim=128,out_point=64,out_variablenum=33):
        
        self.out_point = out_point
        self.out_variablenum = out_variablenum

        super(DeepConvNet, self).__init__()
        
        # Convolution layer 1
        self.conv1 = nn.Conv2d(input_dim, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Convolution layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolution layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Convolution layer 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Output layer 1
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=out_point*2, kernel_size=1, stride=1, padding=0)
        
        # Output layer 2
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=out_point*out_variablenum, kernel_size=1, stride=1, padding=0)
        
        
    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        out1 = self.conv5(x).view(-1,self.out_point,2)*0.01
        out2 = self.conv6(x).view(-1,self.out_point,self.out_variablenum)*0.01
        return out1, out2

class DeepConvNet2(nn.Module):
    def __init__(self,batch_size = 64,out_num=1):
        super(DeepConvNet2,self).__init__()
        self.batchsize = batch_size
        self.con = nn.Sequential(
            nn.Conv2d(3,32,(4,4),(2,4)),
            nn.LeakyReLU(),
            nn.Conv2d(32,64,(4,4),(2,4)),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,(3,3),(2,4)),
            nn.LeakyReLU()
        )

        self.L   = nn.Sequential(
            nn.Linear(128,512),
            nn.LeakyReLU(),
            nn.Linear(512,out_num)
        )
    
    def forward(self, x ):
        x   = x.float().view(-1,3,18,68)
        b   = x.size()[0]
        # print(x.shape)
        # x   = x.permute(0,3,1,2)
        # print(x.shape)
        out = self.con(x)
        out = torch.reshape(out,(b,-1))
        out_con = self.L(out)
        return out_con
    
class DeepConvNet3(nn.Module):
    def __init__(self,input_dim=128,out_point=64,out_variablenum=64):
        
        self.input_dim = input_dim
        self.out_point = out_point
        self.out_variablenum = out_variablenum

        super(DeepConvNet3, self).__init__()
        
        # Convolution layer 1
        self.conv1 = nn.Conv2d(input_dim, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Convolution layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolution layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Convolution layer 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Output layer 1
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=out_point*3,kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # out = self.conv5(x).view(-1,self.out_point,3)*0.05 lase_result_16的解
        # out1 = out[:,:,[1,2]]
        # out2 = out[:,:,[0]]

        out = self.conv5(x).view(-1,self.out_point,3)
        out1 = out[:,:,[1,2]] * 0.1
        out2 = out[:,:,[0]]   * 0.0001

        return out1, out2    

class DeepConvNet_easy(nn.Module):
    def __init__(self,input_dim=128,out_point=64,out_variablenum=64):
        
        self.input_dim = input_dim
        self.out_point = out_point
        self.out_variablenum = out_variablenum

        super(DeepConvNet_easy, self).__init__()
        
        # Convolution layer 1
        self.conv1 = nn.Conv2d(input_dim, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Convolution layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolution layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Convolution layer 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Output layer 1
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=out_point*3,kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # out = self.conv5(x).view(-1,self.out_point,3)*0.05 lase_result_16的解
        # out1 = out[:,:,[1,2]]
        # out2 = out[:,:,[0]]

        out = self.conv5(x).view(-1,self.out_point,3)
        out1 = out[:,:,[1,2]] 
        out2 = out[:,:,[0]] * 0.01
        return out1, out2   

class DeepConvNet_easy2(nn.Module):
    def __init__(self,input_dim=128,out_point=64,out_variablenum=64):
        
        self.input_dim = input_dim
        self.out_point = out_point
        self.out_variablenum = out_variablenum

        super(DeepConvNet_easy2, self).__init__()
        
        # Convolution layer 1
        self.conv1 = nn.Conv2d(input_dim, 32, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Convolution layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolution layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Convolution layer 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        # Output layer 1
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=out_point*3,kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        # out = self.conv5(x).view(-1,self.out_point,3)*0.05 lase_result_16的解
        # out1 = out[:,:,[1,2]]
        # out2 = out[:,:,[0]]

        out = self.conv8(x).view(-1,self.out_point,3)
        out1 = out[:,:,[1,2]]
        out2 = out[:,:,[0]]  

        return out1, out2   
if __name__ == "__main__":    
    
    tmp = torch.randn([128,128])
    M = DeepConvNet3()
    b1,b2 = M(tmp)
    print(b1.shape,b2.shape)
    exit()
    model = DeepConvNet()
    model2 = DeepConvNet2().to('cuda')
    mesh2num_model = ConvNet(32) 

    data_root   = 'F:\\graduate_student\\T2_GANpropeller\\test2\\1_model\\grid_mesh\\'
    data_path   = 'F:\\graduate_student\\T2_GANpropeller\\test2\\1_model\\grid_mesh\\mesh_data_test.csv'

    geom_list = np.load(data_root + 'mesh_data_test.npy')
    mesh = geom_list[23].reshape(18,68,3)

    base_mesh = torch.from_numpy(mesh).float()
    draw_geom(mesh,'base','grid_mesh/pic',-20,15)
    # base_mesh = torch.randn(18,68,3)
    
    process1 = generate_process()
    process1.get_input(base_mesh,mesh2num_model)
    generator = DeepConvNet(process1.input_dim,process1.point_num,process1.vari_Vnum).to(process1.device)
    process1.get_ffd()
    process1.get_PCA()
    process1.get_variation(generator)

    process1.generate_mesh()
    score = model2(process1.batch_mesh)
    print(score.shape)
    
    mesh = process1.batch_mesh.cpu().detach().numpy().reshape(64,18,68,3)
    print(mesh.shape)
    
    for _ in range(mesh.shape[0]):
        draw_geom(mesh[_],str(_),'grid_mesh/pic',-20,15)

