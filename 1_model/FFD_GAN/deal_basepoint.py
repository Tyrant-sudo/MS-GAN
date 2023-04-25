from ffd.deform import get_ffd
import numpy as np

ffd_path = 'F:\\graduate_student\\T2_GANpropeller\\test2\\1_model\\grid_mesh\\avg_mesh_test.npy'
save_path0 = 'ffd/base_point'
save_path1 = 'ffd/base_para'

dim = np.array([3,7,1])
avg_mesh  = np.load(ffd_path)
mesh_plot = avg_mesh.reshape(-1,3)

b,p = get_ffd(mesh_plot,dim)

np.save(save_path0,p)
np.save(save_path1,b)