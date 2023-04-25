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
import draw
from ffd import deform 
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
    return np.dot(DataMat ,K_eigenVector)


# 重构数据
def Reconstruction(lowDataMat, K_eigenVector, meanVal):
    reconDataMat = np.dot(lowDataMat,K_eigenVector.T) + meanVal
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
    b = b
    # noise = np.random.randn(lowDataMat.shape[0],lowDataMat.shape[1] - b)
    # lowDataMat[:,b:] = lowDataMat[:,b:] + noise
    lowDataMat[:,b:] = 0
    reconDataMat = Reconstruction(lowDataMat, V, meanVal.T)
    return reconDataMat

def PCA_vari0(data):

    #全员参与的vari
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

    # K_eigenValue = eigenvalue[-(lower+1):-(upper + 1):-1]
   
    # V = V[:, K_eigenValue]

    return  V, meanVal,dataMat

def PCA_vari1(data,upper = 11):

    #全员参与的vari
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

    K_eigenValue = eigenvalue[-1:-(upper + 1):-1]
   
    V = V[:, K_eigenValue]

    return  V, meanVal,dataMat

def draw_mesh_point(mesh,point,name = 'test',path = 'pic'):
    def draw_geom1(foil_data,ax,alpha=0.7):
    # p = foil_data
        x = foil_data[:,:,0]
        y = foil_data[:,:,1]
        z = foil_data[:,:,2]
       
        ax.plot_surface(x,y,z,rstride = 1,cstride = 1, cmap= 'gist_yarg',alpha = alpha)

        plt.axis('off')

        return ax    
    def draw_ffd1(foil_data,ax):
        # p = foil_data
        x = foil_data[:,0]
        y = foil_data[:,1]
        z = foil_data[:,2]

        ax.scatter3D(x,y,z,s=10,c = 'r')

        return ax
    def draw_round(foil_data,ax):
        # p = foil_data
        x = foil_data[:,:,0].reshape(-1)
        y = foil_data[:,:,1].reshape(-1)
        z = foil_data[:,:,2].reshape(-1)

        ax.plot3D(x,y,z,alpha = 0.5,c='b')

        # plt.axis('off')
        
        # plt.close()

        return ax
    fig = plt.figure(figsize=(15,7))
    ax = plt.axes(projection="3d")
    ax.set_xlim(0,1)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.5,0.5)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)
    ax.view_init(elev=-20, azim=15)

    plt.axis('off')
    ax = draw_geom1(mesh,ax)
    t_point = point.reshape(-1,3)
    ax = draw_ffd1(t_point,ax)
    new_plots = point
    for i in range(new_plots.shape[0]):
        a = new_plots[i,:,:,:]
        a = a.transpose(1,0,2)
        a[0,:,:] = a[0,::-1,:]
        p1  = a[0,0,:]
        p2  = a[-1,-1,:]
        ax = draw_round(a,ax)
        ax.plot3D([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],alpha = 1,c='b')


    plt.savefig(path + '/' + name)
    # plt.show()
    
    plt.close()
    return 
def draw_plot_point(mesh,point,name = 'test',path = 'pic'):
    def draw_geom1(foil_data,ax,alpha=0.7):
    # p = foil_data
        x = foil_data[:,:,0]
        y = foil_data[:,:,1]
        z = foil_data[:,:,2]
       
        ax.scatter3D(x,y,z,c = 'slategrey',s = 0.5,alpha = alpha)

        plt.axis('off')

        return ax    
    def draw_ffd1(foil_data,ax):
        # p = foil_data
        x = foil_data[:,0]
        y = foil_data[:,1]
        z = foil_data[:,2]

        ax.scatter3D(x,y,z,s=10,c = 'r')

        return ax
    def draw_round(foil_data,ax):
        # p = foil_data
        x = foil_data[:,:,0].reshape(-1)
        y = foil_data[:,:,1].reshape(-1)
        z = foil_data[:,:,2].reshape(-1)

        ax.plot3D(x,y,z,alpha = 0.5,c='b')

        # plt.axis('off')
        
        # plt.close()

        return ax
    fig = plt.figure(figsize=(15,7))
    ax = plt.axes(projection="3d")
    ax.set_xlim(0,1)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.5,0.5)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)
    ax.view_init(elev=-20, azim=15)

    plt.axis('off')
    ax = draw_geom1(mesh,ax)
    t_point = point.reshape(-1,3)
    ax = draw_ffd1(t_point,ax)
    new_plots = point
    for i in range(new_plots.shape[0]):
        a = new_plots[i,:,:,:]
        a = a.transpose(1,0,2)
        a[0,:,:] = a[0,::-1,:]
        p1  = a[0,0,:]
        p2  = a[-1,-1,:]
        ax = draw_round(a,ax)
        ax.plot3D([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],alpha = 1,c='b')


    plt.savefig(path + '/' + name)
    # plt.show()
    
    plt.close()
    return 
def ffd_transform(foil_data,dim = (3,7,1),scale = 0.1):
    
    geom = foil_data.reshape(-1,3)

    para,point = deform.get_ffd(geom,dim)

    new_plots = point.reshape(dim[0]+1,dim[1]+1,dim[2]+1,3)
    
    noise = np.random.randn(*(new_plots.shape)) 
    
    new_plots = new_plots + noise * scale
    
    tmp  = new_plots.reshape(-1,3)
    new_mesh = np.dot(para,tmp)

    new_mesh = new_mesh.reshape(*(foil_data.shape))
    return new_mesh,new_plots

def ffd_transform1(foil_data,dim = (3,7,1),scale = 0.1):
    
    geom = foil_data.reshape(-1,3)

    para,point = deform.get_ffd(geom,dim)

    new_plots = point.reshape(dim[0]+1,dim[1]+1,dim[2]+1,3)
    
    noise = np.linspace(-scale,scale,21)
    
    new_meshs  = []
    
    for i in noise:
        
        new_plots = new_plots*(1 + i)
        
        tmp  = new_plots.reshape(-1,3)
        new_mesh = np.dot(para,tmp)

        new_mesh = new_mesh.reshape(*(foil_data.shape))
        new_meshs.append(new_mesh)
    return np.array(new_meshs)

def ffd_transform2(foil_data,dim = (3,7,1),angle = np.pi/4,axis = np.array([1,0,0])):
    
    geom = foil_data.reshape(-1,3)

    para,point = deform.get_ffd(geom,dim)

    new_plots = point.reshape(dim[0]+1,dim[1]+1,dim[2]+1,3)
    
    dis_plots = new_plots[[1],:,:,:].reshape(-1,3)

    # 计算旋转矩阵
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis / np.linalg.norm(axis)
    rot_mat = np.array([[t * x ** 2 + c, t * x * y - s * z, t * x * z + s * y],
                        [t * x * y + s * z, t * y ** 2 + c, t * y * z - s * x],
                        [t * x * z - s * y, t * y * z + s * x, t * z ** 2 + c]])
    rotated_points = np.dot(dis_plots, rot_mat)
    dis_plots = rotated_points.reshape(1,dim[1]+1,dim[2]+1,3)
    new_plots[[1],:,:,:] = dis_plots
    
    tmp  = new_plots.reshape(-1,3)
    new_mesh = np.dot(para,tmp)

    new_mesh = new_mesh.reshape(*(foil_data.shape))

    return new_mesh,new_plots
def ffd_transform3(foil_data,dim = (3,7,1),scale = 0.5):
    
    geom = foil_data.reshape(-1,3)

    para,point = deform.get_ffd(geom,dim)

    new_plots = point.reshape(dim[0]+1,dim[1]+1,dim[2]+1,3)
    
    dis_plots = new_plots[[1],:,:,:].reshape(-1,3)

    dis_plots[:,[1]] = dis_plots[:,[1]] * scale

    dis_plots = dis_plots.reshape(1,dim[1]+1,dim[2]+1,3)
    new_plots[[1],:,:,:] = dis_plots
    
    tmp  = new_plots.reshape(-1,3)
    new_mesh = np.dot(para,tmp)

    new_mesh = new_mesh.reshape(*(foil_data.shape))

    return new_mesh,new_plots
def ffdpca_transform0(foil_data,dim = (3,7,1),upper = 11,lower = 0,scale1 = 0.1,scale2 = 0.1):
# 保守
    geom = foil_data.reshape(-1,3)

    para,point = deform.get_ffd(geom,dim)

    vari1 = np.random.randn(*(point.shape))
    point = point + vari1 * scale1

    vari_Vnum = upper - lower

    V, meanVal,dataMat = PCA_vari0(para,upper,lower)
    
    vari2 = np.random.randn(V.shape[0],upper - lower)
    V[:,lower:upper] = V[:,lower:upper] + vari2*scale2
    
    low_mat    = getlowDataMat(dataMat,V)
    para = Reconstruction(low_mat,V,meanVal)

    new_mesh = np.dot(para,point)

    return new_mesh

def ffdpca_transform00(foil_data,dim = (3,7,1),scale1 = 0.1,scale2 = 0.05):
# 保守
    geom = foil_data.reshape(-1,3)
    total_dim = (dim[0]+1)*(dim[1]+1)*(dim[2]+1)
    para,point = deform.get_ffd(geom,dim)

    vari1 = np.random.randn(*(point.shape))
    point = point + vari1 * scale1


    V, meanVal,dataMat = PCA_vari0(para)
    
    vari2 = np.random.randn(total_dim,1)
    

    V[:,:] = (vari2 *scale2 + 1) * V[:,:]

    low_mat    = getlowDataMat(dataMat,V)
    para = Reconstruction(low_mat,V,meanVal)

    new_mesh = np.dot(para,point)

    return new_mesh

def ffdpca_transform1(foil_data,dim = (3,7,1),upper = 11,lower = 0,scale1 = 0.1,scale2 = 0.1):
# 保守
    geom = foil_data.reshape(-1,3)

    para,point = deform.get_ffd(geom,dim)

    vari1 = np.random.randn(*(point.shape))
    point = point + vari1 * scale1

    vari_Vnum = upper - lower

    V, meanVal,dataMat = PCA_vari0(para,upper,lower)
    
    # vari2 = np.random.randn(V.shape[0],upper - lower)
    vari2 = np.linspace(-scale2,scale2,21)

    new_meshs = []
    V0 =  V[:,lower:upper] 
    for i in vari2:
        V[:,lower:upper] = V0*(1 + i)
        low_mat    = getlowDataMat(dataMat,V)
        para = Reconstruction(low_mat,V,meanVal)

        new_mesh = np.dot(para,point)
        new_meshs.append(new_mesh)
    new_meshs = np.array(new_meshs)
    return new_meshs

def ffdpca_transform2(foil_data,dim = (3,7,1),upper = 11,lower = 0,scale1 = 0.1,scale2 = 0.1):
# 方向性
    geom = foil_data.reshape(-1,3)

    para,point = deform.get_ffd(geom,dim)

    vari1 = np.random.randn(*(point.shape))
    point = point + vari1 * scale1

    vari_Vnum = upper - lower

    V, meanVal,dataMat = PCA_vari0(para,upper,lower)
    
    # vari2 = np.random.randn(V.shape[0],upper - lower)
    vari2 = np.linspace(-scale2,scale2,21)

    new_meshs = []
    V0 =  V[:,lower:upper] 
    for i in vari2:
        V[:,lower:upper] = V0*(1 + i)
        low_mat    = getlowDataMat(dataMat,V)
        para = Reconstruction(low_mat,V,meanVal)

        new_mesh = np.dot(para,point)
        new_meshs.append(new_mesh)
    new_meshs = np.array(new_meshs)
    return new_meshs

from sklearn.manifold import TSNE
def get_tsne(*args):

    lenth = [0]

    for i,j in enumerate(args):

        if i==0:
            g = j
            lenth.append(g.shape[0])
        else:
            g = np.concatenate((g,j),0)
            lenth.append(g.shape[0])

    X_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random', perplexity=3).fit_transform(g)
    
    rlt = []

    for i in range(len(lenth)-1):
        rlt.append(X_embedded[lenth[i]:lenth[i+1],:])

    return rlt
if __name__ == "__main__":

    mean_data  = 'F:\\graduate_student\\T2_GANpropeller\\test5\\0_database\\grid_mesh\\mean_mesh\\mean_data.npy'

    test = np.load(mean_data)

    new_mesh = ffdpca_transform0(test,(3,7,1),64,11,0.01,0.01)

    draw.easy_draw(new_mesh.reshape(-1,68,3))