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
import time
import torch
import torch.autograd as autograd

def fluent(data):
    new_x = np.linspace(0,1,200)
    x     = np.linspace(0,1,data.shape[0])
    f = interpolate.interp1d(x,data,'cubic')
    new_data = f(new_x)
    return new_data

def create_line(x,y,z,ax,chord,R):
    lenth = x.shape[0]
   
    z_mean = np.mean(z)
    y_mean = np.mean(y)
    x_mean = np.mean(x)

    center_z = z_mean - R*3/3
    center_y = y_mean
    center_x = x_mean
    center = np.array([[center_x,center_y,center_z]]).repeat(lenth,0)
    
    theta_left = np.arcsin(chord/R)
    theta_left = 0.002
    theta = np.linspace(-theta_left,theta_left,lenth)

    dis_x = np.zeros(x.shape[0])
    displace = np.array([dis_x,R*np.sin(theta),R*np.cos(theta)]).T

    line  = center +  displace
    
    line = line[::3,:]
    # ax.scatter3D(line[:,0],line[:,1],line[:,2],s = 1,c = 'cyan')

    return line

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

def draw_block(foil_data,name,pic_path):
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

def draw_geom(foil_data,name,pic_path,elev = -65,azim = 10):

    pic_name = pic_path +'/'+ str(name) + '.png'
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection="3d")
    ax.set_xlim(0,1)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.5,0.5)

    ax.view_init(elev=elev, azim=azim)

    outline = []

    lines = []
    for i in range(foil_data.shape[0]):

        x = foil_data[i,:,0]
        y = foil_data[i,:,1]
        z = foil_data[i,:,2]

        y_left  = np.argmin(y)
        y_right = np.argmax(y)

        tmp     = [[x[y_left],y[y_left],z[y_left]],[x[y_right],y[y_right],z[y_right]]]
        p_range = outline.append(tmp)

        ax.scatter3D(x,y,z,s = 5,c = 'tomato')
        line = create_line(x,y,z,ax,np.abs(y[y_right] - y[y_left]),100)
        lines.append(line)
        ax.plot3D(x,y,z, c = 'gray',alpha = 0.5)
        
    lines = np.array(lines)
    outline = np.array(outline)
    x_left = fluent(outline[:,0,0])
    y_left = fluent(outline[:,0,1])
    z_left = fluent(outline[:,0,2])
    
    x_right = fluent(outline[:,1,0])
    y_right = fluent(outline[:,1,1])
    z_right = fluent(outline[:,1,2])

    ax.plot3D(x_left,y_left,z_left, c = 'k',alpha = 0.5)
    ax.plot3D(x_right,y_right,z_right, c = 'k',alpha = 0.5)
    
    plt.axis('off')
    plt.savefig(pic_name)
    # plt.show()
    plt.close()
    return lines
# 数据中心化
def Z_centered(dataMat):
    rows, cols = dataMat.shape
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    meanVal = np.tile(meanVal, (rows, 1))
    newdata = dataMat - meanVal
    return newdata, meanVal


# 最小化降维造成的损失，确定k
def Percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)  # 升序
    sortArray = sortArray[-1::-1]  # 逆转，即降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num


# 得到最大的k个特征值和特征向量
def EigDV(covMat, p):
    
    D, V = np.linalg.eig(covMat)  # 得到特征值和特征向量
    
    k = Percentage2n(D, p)  # 确定k值
    print("保留{}信息，降维后的特征个数：".format(p) + str(k) + "\n")
    eigenvalue = np.argsort(D)
    K_eigenValue = eigenvalue[-1:-(k + 1):-1]
   
    K_eigenVector = V[:, K_eigenValue]

    return K_eigenValue, K_eigenVector


# 得到降维后的数据
def getlowDataMat(DataMat, K_eigenVector):
    return np.dot(DataMat , K_eigenVector)


# 重构数据
def Reconstruction(lowDataMat, K_eigenVector, meanVal):
    reconDataMat = np.dot(lowDataMat , K_eigenVector.T) + meanVal
    return reconDataMat

import decimal
# PCA算法
def PCA(data, p,b=11):
    dataMat = np.float32(np.mat(data))
    # 数据中心化
    dataMat, meanVal = Z_centered(dataMat)
    # 计算协方差矩阵
    
    covMat = np.cov(dataMat)
    # covMat = np.cov(dataMat,rowvar=False)

    # 得到最大的k个特征值和特征向量
    D, V = EigDV(covMat, p)
    
    # 得到降维后的数据
    lowDataMat = getlowDataMat(dataMat.T, V)
    # 重构数据
    # noise = np.random.randn(lowDataMat.shape[0],lowDataMat.shape[1] - b)
    # lowDataMat[:,b:] = lowDataMat[:,b:] + noise
    lowDataMat[:,b:] = 0
    reconDataMat = Reconstruction(lowDataMat, V, meanVal.T)
    return reconDataMat


def PCA_vari(data,b=11):
    dataMat = data
    # 数据中心化
    dataMat, meanVal = Z_centered(dataMat)
    # 计算协方差矩阵
    # covMat = Cov(dataMat)
    # covMat = np.cov(dataMat)
    covMat = np.cov(dataMat,rowvar=False)

    # 得到最大的k个特征值和特征向量
    D, V = np.linalg.eig(covMat)

    eigenvalue = np.argsort(D)
    K_eigenValue = eigenvalue[-1:-(b + 1):-1]
   
    V = V[:, K_eigenValue]


    return  V, meanVal,dataMat


def calc_gradient_penalty(netD, real_data, fake_data):
    
    real_data = real_data.view(-1,fake_data.shape[1],fake_data.shape[2])
    scale = real_data.shape[0]
    fake_data = fake_data[:scale,:,:]
    alpha = torch.rand(real_data.shape[0], 1,1)
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


def easy_save(foil_data, path):
    # p = foil_data
    foil_data = foil_data.reshape(-1,3)
    x = foil_data[:,0]
    y = foil_data[:,1]
    z = foil_data[:,2]

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    # ax.set_xlim(0,1)
    # ax.set_ylim(-0.5,0.5)
    # ax.set_zlim(-0.5,0.5)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)

    ax.scatter3D(x,y,z,s=0.5)
    ax.view_init(elev=75, azim=0)
    plt.axis('off')
    plt.savefig(path)
    plt.close()
    return 

if __name__ =='__main__':

    t = np.random.randn(1224,64)
    V,meanVal,dataMat = PCA_vari(t)

    lowDataMat = getlowDataMat(dataMat, V)
    reconDataMat = Reconstruction(lowDataMat, V, meanVal)

    print(reconDataMat.shape)