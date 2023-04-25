import numpy as np

import model
from model import MyDataSet

import torch
import torch.utils.data as Data

from model import MyDataSet,generate_process,process2

if __name__ =="__main__":
    data_root   = 'F:\\graduate_student\\T2_GANpropeller\\test5\\0_database\\mixed_model\\'
    save_path1   = 'result/process/'
    save_path2   = 'result/last_result/'
    save_path3   = 'result/last_result/'
    geom_list = np.load(data_root + 'easy_real.npy')

    sample_num= 3
    device     = 'cuda'

    noise_dim  = 8
    mesh2num   = 8
    BATCH_SIZE = 256
    batch_size = BATCH_SIZE
    ffd_dim    = (3,7,1)
    point_num  = (ffd_dim[0]+1)*(ffd_dim[1]+1)*(ffd_dim[2]+1)
    dim_mesh   = (18,68,3)
    upper_vdim = 64
    lower_vdim = 11

    train = MyDataSet(geom_list)
    train_loader = Data.DataLoader(train,BATCH_SIZE,True)
    
    mesh2num_model = model.ConvNet(mesh2num).to(device) 
    G = model.DeepConvNet_easy2(noise_dim + mesh2num,point_num,upper_vdim).to(device)
    D = model.DeepConvNet2(batch_size,1).to(device) #输出为1维向量
    
    mesh2num_model.load_state_dict(torch.load(save_path3 +"m2n_param.pkl"))
    G.load_state_dict(torch.load(save_path3 +"generator_param.pkl"))
    D.load_state_dict(torch.load(save_path3 +"discriminator_param.pkl"))

    lrG = 0.1
    lrD = 0.0004
    train_epoch = 3000
    ave_num = 10

    process1 = generate_process(batch_size,noise_dim,mesh2num,device,lower_vdim,upper_vdim)
    p2 = process2(process1,mesh2num_model,G,D,sample_num,train_epoch,lrG,lrD,save_path1,save_path2)
    G = D = mesh2num_model = []

    base_line = np.mean(geom_list,0)
    base_line = torch.from_numpy(base_line)
    # exit()
    p2.sequence = [500,1500,2500]
    for epoch in range(1781,train_epoch):

        p2.train_one_epoch(epoch, train_loader ,1)

        if epoch%10 == 0:
            p2.easy_draw(epoch)
