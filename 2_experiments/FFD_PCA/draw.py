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
from ffd import deform
import math

import open3d as o3d
from scipy.spatial import ConvexHull
import imageio.v2 as imageio
from scipy.interpolate import splev, splprep, interp1d
from scipy.integrate import cumtrapz

data_root_path   = 'F:\\graduate_student\\T2_GANpropeller\\test2\\1_model\\grid_mesh\\'
def get_group(name_list,save_path,pic_width,pic_heigh,figsize=(20,6)):
    column,row = len(name_list),len(name_list[0])
    
    fig = plt.figure(figsize=figsize)

    for i in range(column):
        for j in range(row):

            posi = j + i * row + 1
          
            ax = fig.add_subplot(column,row,posi)
            ax.axis('off')
            im   = img.imread(name_list[i][j])
            s = im.shape
            left  = (s[0] - pic_width)//2
            right = (s[0] + pic_width)//2
            below = (s[1] - pic_heigh)//2 
            upper = (s[1] + pic_heigh)//2

            im   = im[left:right ,below:upper,:]
            ax.imshow(im)
    plt.savefig(save_path)
    plt.show()
    plt.close()
    
def get_gif(path,save_name):
    all_files = os.listdir(path)

    # 仅保留以".png"结尾的文件
    png_files = [file for file in all_files if file.endswith('.png')]
    images = []

    for i in range(len(png_files)):
        images.append(imageio.imread(path + png_files[i]))

    imageio.mimsave(path + save_name, images, duration=0.1)

def data_root():
    """
    name1 = 'mesh_data_modified.npy' (44467,18,68,3)\n
    name2 = 'mesh_data_test.npy' (4452,18,68,3)
    """

    return data_root_path

def fluent(data):
    new_x = np.linspace(0,1,200)
    x     = np.linspace(0,1,data.shape[0])
    f = interpolate.interp1d(x,data,'cubic')
    new_data = f(new_x)
    return new_data
def interpolate_sec(Q, N=200, k=3, D=20, resolution=1000):
    ''' Interpolate N points whose concentration is based on curvature. '''
    Q = np.concatenate((Q,Q[0][None,:]),0)
    res, fp, ier, msg = splprep(Q.T, u=None, k=k, s=1e-6, per=0, full_output=1)
    tck, u = res
    uu = np.linspace(u.min(), u.max(), N)
    x_new, y_new = splev(uu, tck, der=0)
    xy_new = np.concatenate((x_new[:,None], y_new[:,None]),1)
    return xy_new

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


def draw_geom_line(foil_data,name,pic_path,elev = -65,azim = 10):

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

        ax.scatter3D(x,y,z,s = 5,c = 'tomato',alpha = 0.5)
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

def easy_draw(foil_data):
    # p = foil_data
    x = foil_data[:,:,0]
    y = foil_data[:,:,1]
    z = foil_data[:,:,2]

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.set_xlim(0,1)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.5,0.5)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)

    ax.scatter3D(x,y,z,s=0.5)

    plt.axis('off')
    plt.show()
    plt.close()
    return 


def draw_surface(foil_data,name,pic_path):
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

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)

    ax.plot_surface(x,y,z,cmap= 'ocean')

    plt.axis('off')
    plt.savefig(pic_name)
    plt.close()


def draw_ffd(foil_data,name,path,point = np.array([])):
    def draw_geom1(foil_data,ax,alpha=0.3):
    # p = foil_data
        x = foil_data[:,:,0]
        y = foil_data[:,:,1]
        z = foil_data[:,:,2]

        ax.plot_surface(x,y,z,color='#3d7afd',alpha = alpha)

        plt.axis('off')

        return ax
    def draw_mesh1(foil_data,ax,alpha=0.3):
    # p = foil_data
        x = foil_data[:,:,0]
        y = foil_data[:,:,1]
        z = foil_data[:,:,2]

        ax.scatter3D(x,y,z,color='b',alpha = alpha,s=0.05)

        plt.axis('off')

        return ax
    def draw_ffd1(foil_data,ax):
        # p = foil_data
        x = foil_data[:,0]
        y = foil_data[:,1]
        z = foil_data[:,2]

        ax.scatter3D(x,y,z,s=2,c = 'r')

        
        # plt.show()
        # plt.close()

        return ax
    def draw_round(foil_data,ax):
        # p = foil_data
        x = foil_data[:,:,0].reshape(-1)
        y = foil_data[:,:,1].reshape(-1)
        z = foil_data[:,:,2].reshape(-1)

        ax.plot3D(x,y,z,alpha = 0.5,c='gray')

        # plt.axis('off')
        
        # plt.close()

        return ax
    def draw_line(foil_data,ax):
        # p = foil_data
        x = foil_data[:,0].reshape(-1)
        y = foil_data[:,1].reshape(-1)
        z = foil_data[:,2].reshape(-1)

        ax.plot3D(x,y,z,alpha = 0.5,c='gray')

        # plt.axis('off')
        
        # plt.close()

        return ax
    
    present_geom = foil_data
    dim = (3,7,1)
    _ = present_geom.reshape(-1,3)
    para,points = deform.get_ffd(_,dim)
    
    if point.shape[0] == 0:
        point = points

    fig = plt.figure(figsize=(15,7))
    ax = plt.axes(projection="3d")
    ax.set_xlim(0,1)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.5,0.5)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)
    plt.axis('off')
    ax = draw_geom1(present_geom,ax)
    ax = draw_mesh1(present_geom,ax)
    ax = draw_ffd1(point,ax)
    new_plots = point.reshape(4,8,2,3)
    for i in range(new_plots.shape[0]):
        a = new_plots[i,:,:,:]
        a = a.transpose(1,0,2)
        a[0,:,:] = a[0,::-1,:]
        p1  = a[0,0,:]
        p2  = a[-1,-1,:]
        ax = draw_round(a,ax)
        ax.plot3D([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],alpha = 0.5,c='gray')

    for j in range(new_plots.shape[1]):
        for k in range(new_plots.shape[2]):
            if j%2==0:
                b = new_plots[:,j,k,:]
                # ax = draw_line(b,ax)

    radius = 0.17
    thickness  = 0.05

    dim = (18, 68, 3)
    tmp = np.linspace(0,2*np.pi,30)
    ttmp = np.linspace(radius,0,50)

    for i,j in enumerate( ttmp):
        x = j * np.cos(tmp)[None,:]
        y = j * np.sin(tmp)[None,:]
        z0 = np.array([thickness for _ in range(30)])[None,:]
        z1 = np.array([-thickness for _ in range(30)])[None,:]
        create_geom = np.concatenate((x,y,z0),0).T[None,:]
        # b = np.concatenate((x,y,z1),0).T[None,:]
        # create_geom = np.concatenate((a,b),0)
        if i ==0:
            create_geoms = create_geom
        else:
            create_geoms = np.concatenate((create_geoms,create_geom),0)
    create_geoms0 = create_geoms.transpose(1,0,2)
    for i,j in enumerate( ttmp):
        x = j * np.cos(tmp)[None,:]
        y = j * np.sin(tmp)[None,:]
        z0 = np.array([thickness for _ in range(30)])[None,:]
        z1 = np.array([-thickness for _ in range(30)])[None,:]
        create_geom = np.concatenate((x,y,z1),0).T[None,:]
        # b = np.concatenate((x,y,z1),0).T[None,:]
        # create_geom = np.concatenate((a,b),0)
        if i ==0:
            create_geoms = create_geom
        else:
            create_geoms = np.concatenate((create_geoms,create_geom),0)
    create_geoms1 = create_geoms.transpose(1,0,2)

    x = radius * np.cos(tmp)[None,:]
    y = radius * np.sin(tmp)[None,:]
    z0 = np.array([thickness for _ in range(30)])[None,:]
    z1 = np.array([-thickness for _ in range(30)])[None,:]
    a = np.concatenate((x,y,z0),0).T[None,:]
    b = np.concatenate((x,y,z1),0).T[None,:]
    create_geoms2 = np.concatenate((a,b),0)

    # create_geoms = np.concatenate((create_geoms0,create_geoms1),1)
    ax = draw_geom1(create_geoms1,ax,alpha=0.8)
    ax = draw_geom1(create_geoms0,ax,alpha=0.8)

    ax = draw_geom1(create_geoms2,ax,alpha=0.8)

    reversed_geoms = -present_geom

    ax = draw_geom1(reversed_geoms,ax,alpha=0.8)
    plt.savefig(path + '/' + name)
    plt.show()
    
    plt.close()
    return point


def distance(p1, p2):
    """
    计算两点之间的距离
    """
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def angle(p1, p2):
    """
    计算极角
    """
    return math.atan2(p2[1]-p1[1], p2[0]-p1[0])


def distance(p1, p2):
    """
    计算两点之间的距离
    """
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def angle(p1, p2):
    """
    计算两点之间的极角
    """
    return math.atan2(p2[1]-p1[1], p2[0]-p1[0])

def clockwise_sort(points):
    """
    按照从近到远且顺时针的顺序排序点集
    """
    # 计算中心点
    center_x = sum([p[0] for p in points]) / len(points)
    center_y = sum([p[1] for p in points]) / len(points)
    center = (center_x, center_y)

    # 计算每个点到中心点的距离和极角
    polar_points = [(p, distance(p, center), angle(center, p)) for p in points]

    # 按照极角排序
    polar_points = sorted(polar_points, key=lambda x: x[2])

    # 如果存在多个点共线的情况，按照距离排序
    for i in range(len(polar_points)-1):
        p1, dist1, angle1 = polar_points[i]
        p2, dist2, angle2 = polar_points[i+1]
        if angle1 == angle2 and dist1 > dist2:
            polar_points[i], polar_points[i+1] = polar_points[i+1], polar_points[i]

    # 返回排序后的点集
    a = [list(p[0]) for p in polar_points]
    
    return np.array(a)

def get_value_center(data):

    """
    得到体积和质心
    """


    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(data)


    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    alpha = 0.05

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()
    is_watertight = mesh.is_watertight()
    is_inter      = mesh.is_self_intersecting()

    try:
        volume = mesh.get_volume()
        center = mesh.get_center()
    except:
        return is_inter,False,-1,-1
    return is_inter,is_watertight,volume,center

def get_section(data,posi = [0.5]):
    """
    从点云数据生成某些截面的数据
    """


    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(data)


    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    alpha = 0.05

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()
    mesh0 = mesh.sample_points_uniformly(500000)
    

    points= np.asarray(mesh0.points)
    
    for i,_ in enumerate(posi):
        po = np.argwhere(np.abs(points[:,0]-_) < 0.001)
        if po.shape[0] <2:
            continue
        p = points[po].reshape(-1,3)
        
        x = fluent(p[:,0])[:,None]
        x = np.ones_like(x) * _

        yz = clockwise_sort(p[:,[1,2]])

        yz = interpolate_sec(yz)
        
        p1 = np.concatenate((x,yz),1)
            
        try:
            result = np.concatenate((result,p1[None,:]),0)
        except:
            result = p1[None,:]

    return result

from scipy import spatial
def draw_section(foil_data,sec_po = [0.3,0.5],name = '',pic_path = '',elev=-20, azim=15,scale = False,displace = 0.5,rota = True):
    
    def farthest_points(points):
        
        pts = points
        
        # two points which are fruthest apart will occur as vertices of the convex hull
        candidates = pts[spatial.ConvexHull(pts).vertices]
        
        # get distances between each pair of candidate points
        dist_mat = spatial.distance_matrix(candidates,candidates)
        
        # get indices of candidates that are furthest apart
        i,j = np.unravel_index(dist_mat.argmax(),dist_mat.shape)
        
        a,b = np.argwhere(points == candidates[i])[0][0],np.argwhere(points==candidates[j])[0][0]
        
        if points[a][0] < points[b][0]:
            return a,b
        else:
            return b,a

    def norm_sec(section,scale,displace):
        
        x  = section[:,0][:,None]
        y  = section[:,1]
        z  = section[:,2]
        points = section[:,[1,2]]
        
        p0,p1 = farthest_points(points)
        # p0 = np.argmin(y)
        # p1 = np.argmax(y)
        points[:,1] = points[:,1] - points[p0,1] 
        points[:,0] = points[:,0] - points[p0,0]
        if scale == True:
            scale = 1/np.sqrt((z[p1]-z[p0])**2 + (y[p1]-y[p0])**2)
        else:
            scale = 1.5
        
        theta = -math.atan2(z[p1]-z[p0],y[p1]-y[p0])
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        
        scale_matrix = np.array([[scale, 0], [0, scale]])

        if rota:
            rotated_points = np.dot(rot_matrix, points.T).T
        else:
            rotated_points = points
        scaled_points = np.dot(scale_matrix, rotated_points.T).T

        points = np.concatenate((x,scaled_points),1)
        points[:,1] = points[:,1] - displace
        
        return points
    
    rlt = []
    foil_data1 = get_section(foil_data,sec_po)
    for _ in foil_data1:
        
        standard = norm_sec(_,scale,displace)
        rlt.append(standard)
                
    
    rlt = np.array(rlt)

    pic_name = pic_path +'/'+ str(name) + '.png'
    
    foil_data = foil_data.reshape(-1,68,3)
    x  = foil_data[:,:,0]
    y  = foil_data[:,:,1]
    z  = foil_data[:,:,2]

    
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    ax.set_xlim(0,1)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-0.5,0.5)
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)

    for _ in rlt:
        x1 = _[:,0]
        y1 = _[:,1]
        z1 = _[:,2]
        ax.plot3D(x1,y1,z1,c= 'navy')
    ax.plot_surface(x,y,z,cmap= 'gist_yarg',alpha = 0.7)
    plt.axis('off')
    plt.savefig(pic_name)
    # plt.show()
    plt.close()
    return rlt

def draw_list(crlist,ax,color = 'b'):
    
    hull = spatial.ConvexHull(crlist, qhull_options="QJ")
    poly = plt.Polygon(crlist[hull.vertices, :],alpha = 0.1, color = color)
    ax.add_patch(poly)

def get_data(g):
    
    crlist,betalist = g[:,[0,1]],g[:,[2,3]]
    
    return crlist,betalist

def draw_multi_tsne_distribution(g1 ,g1_space = 8,g2_space = 1, *args):
    """
    g1:[质心距离平均的距离,体积,tsne0,tsne1] crlist,betalist对于数据集
    g2:[质心距离平均的距离,体积,tsne0,tsne1] 对于
    """
    fig = plt.figure(figsize=(20,8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.set_xlim(-0.01,0.04)
    ax1.set_ylim(-1,15)
    g = g1
    g1_l = g1.shape[0]
    g2_l = [g1_l]
    for i,g2 in enumerate(args):
        g = np.concatenate((g,g2),axis=0)
        tmp =  g2.shape[0] + g2_l[i]
        g2_l.append(tmp)

    crlist_all,betalist_all = get_data(g)
    
    if g1!= []:
        
        crlist,betalist = crlist_all[:g1_l,:],betalist_all[:g1_l,:]
        mean_y = np.mean(crlist[:,1])
        # ax1.axhline(y = mean_y,ls = '--',c = 'skyblue')
        crlist,betalist = crlist[::g1_space,:],betalist[::g1_space,:]
        

        ax1.scatter(crlist[:,0],crlist[:,1],c = 'skyblue',alpha =0.5,label = 'database')
        ax2.scatter(betalist[:,0],betalist[:,1],c = 'skyblue',alpha =0.5,label='database')

        # draw_list(crlist,ax1,color='skyblue')
        draw_list(betalist,ax2,color='skyblue')    



    color_bar = ['salmon','mediumpurple','lime']
    c = -1

    for i in range(len(g2_l)-1):

        c+=1
        color = color_bar[c]
        crlist,betalist = crlist_all[g2_l[i]:g2_l[i+1],:],betalist_all[g2_l[i]:g2_l[i+1],:]
        mean_y = np.mean(crlist[:,1])
        
        # ax1.axvline()

        crlist,betalist = crlist[::g2_space,:],betalist[::g2_space,:]


        ax1.scatter(crlist[:,0],crlist[:,1],c = color,alpha =0.5,label = ' ')
        ax2.scatter(betalist[:,0],betalist[:,1],c = color,alpha =0.5,label=' ')

        # draw_list(crlist,ax1,color=color)
        try:
            draw_list(betalist,ax2,color=color) 
            ax1.axhline(y = mean_y,ls = '--',c = color)  
        except:
            ax1.scatter(crlist[:,0],crlist[:,1],marker='*', c = color,alpha =1,s=100)
            ax2.scatter(betalist[:,0],betalist[:,1],marker='*', c = color,alpha =1,s=100)

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    plt.show()
    plt.close()


def draw_mean_distribution(g1 = [],tsne = [],g2 = [],g1_space = 1):
    """
    g1:[质心距离平均的距离,体积]
    g2:[]
    """
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
   
    # ax1.set_ylim(-2.2,-1)
    # ax2.set_xlim(1,20)
    if g1!= []:
        for j,i in enumerate(g1):
            if j%g1_space==0:

                mean_rc = g1[j,0]
                mean_c  = g1[j,1]

                tsne_x = tsne[j,0]
                tsne_y = tsne[j,1]

                color = 'mediumblue'

                if j == 0:
                    ax1.scatter(mean_rc,mean_c,c = color,alpha =1,label='database')
                    ax2.scatter(tsne_x,tsne_y,c=color,alpha =1,label='database')
                else:
                    ax1.scatter(mean_rc,mean_c,c = color,alpha =0.04)
                    ax2.scatter(tsne_x,tsne_y,c=color,alpha =0.04)

    # ax1.legend()
    # ax2.legend()
    plt.show()
    plt.close()


if __name__ == "__main__":

    mean_point = np.mean(np.load(data_root() + 'mesh_data_test.npy'),0)
    # draw_ffd(mean_point,'','')
    
    test_mesh = mean_point.reshape(-1,3)
    p = draw_section(test_mesh,[0.2,0.4,0.6,0.8],'test','pic')
    # print(p.shape)
    # easy_draw(p)
    