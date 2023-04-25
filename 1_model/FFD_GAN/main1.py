import time
import os 
import numpy as np
import csv
import ast
import sys
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pickle
import pandas as pd

import model
from model import MyDataSet

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

cur_path    = os.path.abspath(__file__)
data_root   = 'F:\\graduate_student\\T2_GANpropeller\\test2\\1_model\\grid_mesh\\'
data_path   = 'F:\\graduate_student\\T2_GANpropeller\\test2\\1_model\\grid_mesh\\mesh_data_test.csv'
pic_path    = 'F:\\graduate_student\\T2_GANpropeller\\test2\\1_model\\mymodel\\model_GAN\\result\\'
output_path    = 'F:\\graduate_student\\T2_GANpropeller\\test2\\1_model\\mymodel\\model_GAN\\data_output\\'
header = ('name','coordinate')

device = 'cuda'

def form_Dataset(file_name,scale):
        data = pd.read_csv(file_name)
        list = data.values.tolist()
        cordi = []
        for j,i in enumerate( list):
            temp = ast.literal_eval(i[1])
            if np.max(temp)>2 or np.min(temp)<-2:
                print(j,'err!!!')
                continue
            else:
                cordi.append(temp)
            if j > scale:
                break
        cordi=np.array(cordi)
        return cordi


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def draw_geom(foil_data,name):
    # p = foil_data
    x = foil_data[:,:,0]
    y = foil_data[:,:,1]
    z = foil_data[:,:,2]
    pic_name = pic_path +'/'+ str(name) + '.png'
    
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.set_xlim(0,1)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.5,0.5)
    ax.scatter3D(x,y,z,s=0.05)
    plt.savefig(pic_name)
    plt.close()

def get_cR_beta(coordinate_list,R=1):
    
    r_R       = []
    c_R_list  = []
    beta_list = []
    for i in coordinate_list:
        
        x = i[:,0]
        y = i[:,1]
        z = i[:,2]

        r_R.append(x[0])
        l_max = np.argmax(y)
        l_min = np.argmin(y)
        y_max = y[l_max]
        z_max = z[l_max]
        y_min = y[l_min]
        z_min = z[l_min]
        
        c_R  = np.sqrt((y_max-y_min)**2 + (z_max - z_min)**2)/R
        beta = np.arctan((z_max-z_min)/(y_max-y_min))/np.pi*180
        c_R_list.append(c_R)
        beta_list.append(beta)

    return r_R,c_R_list,beta_list


def draw_block(foil_data,name):
    r_R,c_R,beta = get_cR_beta(foil_data)
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111)
    ax.plot(r_R,beta,label='beta',color='r')
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(r_R,c_R,label='c/R')
    ax2.legend()
    print(name ,'max c:',np.max(c_R),'max beta:',np.max(beta))
    pic_name = pic_path +'/'+ str(name) + '_list.png'
    plt.savefig(pic_name)
    plt.close()


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.shape[0], 1,1,1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

  
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    # TODO: Make ConvBackward diffentiable
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(), 
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() 
    return gradient_penalty

if __name__ == '__main__':
    #生成相对控制点的位移
    if not os.path.exists(data_root + 'mesh_data_test.npy'):
        geom_list = form_Dataset(data_path,100000)
        np.save(data_root+'mesh_data_test',geom_list)
    else:
        geom_list = np.load(data_root + 'mesh_data_modified.npy')

    print(geom_list.shape)

    noise_dim = 12
    # label_dim = 2

    out_dim  = (4,8,2)
    
    z_dimension = noise_dim
    BATCH_SIZE,N_IDEAS = 256,20

    train = MyDataSet(geom_list)
    train_loader = Data.DataLoader(train,BATCH_SIZE,True)
    save_path0 = 'ffd/base_point.npy'
    save_path1 = 'ffd/base_para.npy'
    base_point = np.load(save_path0)
    shape_point = base_point.shape
    base_para  = np.load(save_path1)
    shape_para   = base_para.shape


    # print(base_point.shape)
    # exit()

    base_mesh  = np.dot(base_para,base_point).reshape(18,68,3)
    # draw_geom(base_mesh,2)
    # draw_block(base_mesh,2)
    base_point = torch.tensor(base_point).unsqueeze(0).to(device)
    base_para  = torch.tensor(base_para).unsqueeze(0).to(device)
    base_mesh  = torch.tensor(base_mesh).unsqueeze(0).to(device)

    torch.manual_seed(1234)
    G = model.generator5(noise_dim).to(device)
    D = model.discriminator5().to(device)

    # G.weight_init(100,0.1)
    # D.weight_init(0,0.01)
    # G.load_state_dict(torch.load(output_path +"\\generator_param300.pkl"))
    # D.load_state_dict(torch.load(output_path +"\\discriminator_param300.pkl"))
    # G = cGAN.generator1_2().to(device)
    one = torch.FloatTensor([1])
    mone = one * -1
    if 1:
        one = one.cuda()
        mone = mone.cuda()
    batch_size = BATCH_SIZE
    lrG = 0.0001
    lrD = 0.0004
    train_epoch = 3000

    G_optimizer = optim.RMSprop(G.parameters(),lr = lrG)
    D_optimizer = optim.RMSprop(D.parameters(),lr = lrD)

    result_path = './result'

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    print('training start!')
    start_time = time.time()
    
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        D_real   = []
        D_fake   = []

        G.train()
        D.train()
        if (epoch+1) == int(train_epoch/5):
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        if (epoch+1) == int(train_epoch/2):
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        epoch_start_time = time.time()

        for x_ in train_loader:
            for p in D.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            D_optimizer.zero_grad()  # 判别器D的梯度归零
            x_ = x_.to(device)
            x_ = Variable(x_)
            input = x_
            input_v = autograd.Variable(input)

            z = Variable(torch.randn(x_.shape[0], noise_dim).to(device))
            
            real_out = D(input_v)
           
            d_loss_real = torch.mean(real_out)
            (-d_loss_real).backward()
            
            fake_x = G(z,base_point,base_para)  # 将向量放入生成网络G生成一张图片
            fake_x = autograd.Variable(fake_x)
            fake_out = D(fake_x)

            d_loss_fake = torch.mean(fake_out)
            (d_loss_fake).backward()
            
            gradient_penalty = 10 * calc_gradient_penalty(D, input_v.data, fake_x.data)
            gradient_penalty.backward()
            d_loss = - d_loss_real + d_loss_fake + gradient_penalty
            
            # d_loss.backward()  # 反向传播
            D_optimizer.step() 

            for p in D.parameters():
                p.requires_grad = False

            D_losses.append(d_loss.item())
            D_real.append(d_loss_real.item())
            D_fake.append(d_loss_fake.item())


            for j in range(2):
                G_optimizer.zero_grad()  # 生成器G的梯度归零
                fake_img = G(z,base_point,base_para)  # 将向量放入生成网络G生成一张图片

                defort   = G.defort
                target   = torch.zeros_like(defort)
                R3       = 100*(nn.MSELoss()(defort,target).mean())**2
                R3.backward(retain_graph=True)
               
                # defort     = fake_img
                # target     = base_mesh.repeat(x_.shape[0],1,1,1).float()
                # R3 =  100*(nn.MSELoss()(defort,target).mean())**2             
                # R3.backward(retain_graph=True)

                output = D(fake_img)  # 经过判别器得到结果
                g_loss = - torch.mean(output)
                # bp and optimize
                
                g_loss.backward()  # 反向传播
                G_optimizer.step()  # 更新生成器G参数

                G_losses.append(g_loss.item())
                # for pa in G.con.parameters():
                #     print(pa)
                #     print(R3)
                #     exit()

        if(epoch%10) == 0:
            
            # draw_geom(x_[0].cpu().detach().numpy(),epoch)
            # draw_block(x_[0].cpu().detach().numpy(),epoch)
            draw_geom(fake_x[0].cpu().detach().numpy(), epoch)
            draw_block(fake_x[0].cpu().detach().numpy(), epoch)
            torch.save(G.state_dict(), output_path +"\\generator_param" + ".pkl")
            torch.save(D.state_dict(), output_path +"\\discriminator_param" + ".pkl")

            
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time


        print('[%d/%d] - ptime: %.2f, loss_d: %.3f,loss_d_real:  %.3f,loss_d_fake:   %.3f, loss_g: %.3f, R3 = %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, 
                                                                torch.mean(torch.FloatTensor(D_losses)),
                                                                torch.mean(torch.FloatTensor(D_real)),
                                                                torch.mean(torch.FloatTensor(D_fake)),
                                                                torch.mean(torch.FloatTensor(G_losses)),
                                                                R3))

        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    print("Training finish!... save training results")
    torch.save(G.state_dict(), output_path + "\\generator_param.pkl")
    torch.save(D.state_dict(), output_path + "\\discriminator_param.pkl")

    with open('result/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path='result/train_process.png')