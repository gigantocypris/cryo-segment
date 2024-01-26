# -*- coding: utf-8 -*-

import pickle as pk
import numpy as np
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mrcfile
import os 
iffigv=True


sigma='new'


#save_list=['data_zero','X_new','mean','mean_mat']
save_list=['mean_mat']
for par in save_list:
    f = open(par,'rb')
    locals()[par]=pk.load(f)
    f.close()
    
    
mrc_out=np.zeros(mean_mat.shape)    
cc=0
temp=np.where(np.abs(mean_mat)<=cc)
mrc_out[temp[0],temp[1],temp[2]]=200
temp=np.where(np.abs(mean_mat)==0)
mrc_out[temp[0],temp[1],temp[2]]-=100


 
image_path='mean_surface_'+str(cc)+'.mrc'
if os.path.exists(image_path):
    mrc2=mrcfile.open(image_path,mode='r+')
else:
    mrc2=mrcfile.new(image_path)
mrc2.set_data(mrc_out.astype(np.uint8))
mrc2.close()

mrc_out=np.zeros(mean_mat.shape)    
cc=3
temp=np.where(np.abs(mean_mat)<=cc)
mrc_out[temp[0],temp[1],temp[2]]=200
temp=np.where(np.abs(mean_mat)==0)
mrc_out[temp[0],temp[1],temp[2]]-=100


 
image_path='mean_surface_'+str(sigma)+'_'+str(cc)+'.mrc'
if os.path.exists(image_path):
    mrc2=mrcfile.open(image_path,mode='r+')
else:
    mrc2=mrcfile.new(image_path)
mrc2.set_data(mrc_out.astype(np.uint8))
mrc2.close()

z=95
mrc_out2=np.zeros(mean_mat.shape)
mrc_out2[z:,:,:]=mrc_out[z:,:,:]

image_path='mean_surface_'+str(sigma)+'_'+str(cc)+'_z_big'+str(z)+'.mrc'
if os.path.exists(image_path):
    mrc2=mrcfile.open(image_path,mode='r+')
else:
    mrc2=mrcfile.new(image_path)
mrc2.set_data(mrc_out2.astype(np.uint8))
mrc2.close()

mrc_out2=np.zeros(mean_mat.shape)
mrc_out2[:z,:,:]=mrc_out[:z,:,:]

image_path='mean_surface_'+str(sigma)+'_'+str(cc)+'_z_small'+str(z)+'.mrc'
if os.path.exists(image_path):
    mrc2=mrcfile.open(image_path,mode='r+')
else:
    mrc2=mrcfile.new(image_path)
mrc2.set_data(mrc_out2.astype(np.uint8))
mrc2.close()


   