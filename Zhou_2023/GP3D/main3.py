# -*- coding: utf-8 -*-

from numpy import linalg as LA
import numpy as np
import random
from matplotlib import pyplot as plt
import  scipy.stats as st
from mpl_toolkits.mplot3d import Axes3D
import math
import cv2
import  scipy.stats as st
import time
import mrcfile
import os
image_path='data_all_0.1_0.01.mrc'

image_path='../../5.RL-connect-slice/t2/img_fix_show_8_dilate2.mrc'
image_path='../../5.RL-connect-slice/t2/img_fix_show_7.mrc'
mrc1=mrcfile.open(image_path)
data=mrc1.data

kernel_name='paper3D'
par=[140]
def Cov_mat(X,X2=None,name=None,par=None):
    if isinstance(X2,type(None)):
        X2=X
    r_mat=np.sqrt(np.square(np.expand_dims(X[:,0],1)-np.transpose(np.expand_dims(X2[:,0],1)))\
                 +np.square(np.expand_dims(X[:,1],1)-np.transpose(np.expand_dims(X2[:,1],1)))\
                 +np.square(np.expand_dims(X[:,2],1)-np.transpose(np.expand_dims(X2[:,2],1))))

    r_mat[r_mat==0]=1e-10
    ans=kernelfun(r_mat,name)
    return(ans)

def del_repeat(X,px):
    y_index=X[:,0]*px[1]*px[2]+X[:,1]*px[2]+X[:,2]
    y_index=list(set(y_index))
    y_index.sort()
    n=len(y_index)
    X_new=np.zeros((n,3))
    X_new[:,0]=y_index//(px[1]*px[2])
    y_index=y_index-X_new[:,0]*px[1]*px[2]
    X_new[:,1]=y_index//px[2]
    y_index=y_index-X_new[:,1]*px[2]
    X_new[:,2]=y_index
    X_new=X_new.astype(np.int)
    return(X_new)
    
    
def kernelfun(r_mat,name=None,par=None):
    if name=='paper':
        r2_mat=np.square(r_mat)
        ans=2*(r2_mat)*np.log(r_mat)-(1+2*math.log(R))*r2_mat+R**2
        return(ans)
    if name=='gaussian':
        r2_mat=np.square(r_mat)
        if par is None:
            par=[px/10]
        ans=np.exp(-r2_mat/(2*(par[0]**2)))
        return(ans)
    if name=='paper3D':
        ans=2*np.power(r_mat,3)+3*R*(np.square(r_mat))+R**3
        return(ans)
    return(None)    
        
iffigv=True

random.seed(10)
np.random.seed(10)



   
ind_list=np.unique(data[data>0])
#ind_list=[1,2,3,5,6,7]
# ind_list=[1,6]

mean_mat=np.zeros(data.shape,dtype=np.float64)+1000
#bias_c=np.array([30,50,50])
#mean_mat=mrc_out.astype(np.uint8) 
for ind in ind_list:
    px=np.array(data.shape)
    print('working on ',ind)
    temp=np.where(data==ind)
    data_zero=np.zeros((len(temp[0]),3))
    data_zero[:,0]=temp[0].flatten()
    data_zero[:,1]=temp[1].flatten()
    data_zero[:,2]=temp[2].flatten()
    
    
    data_zero_select=data_zero[data_zero[:,1]==int(np.mean(data_zero[:,1])),:]
    ellipse1 = cv2.fitEllipse(np.array(data_zero_select[:,[0,2]]).astype(np.int))
    
    plt.figure()
    imshow=np.zeros((px[2],px[0]))
    imshow=cv2.ellipse(imshow,ellipse1,255,thickness=1)
    imshow=np.rot90(np.fliplr(imshow),1)
    plt.imshow(imshow)
    plt.scatter(data_zero_select[:,2],data_zero_select[:,0])
    plt.show()
    
    
    temp_loc_1=np.where(imshow>0)
    temp_1_max=np.max(temp_loc_1[0])
    temp_1_min=np.min(temp_loc_1[0])
    
    
    data_zero_select=data_zero[data_zero[:,2]==int(np.mean(data_zero[:,2])),:]
    ellipse2 = cv2.fitEllipse(np.array(data_zero_select[:,[0,1]]).astype(np.int))
    
    plt.figure()
    imshow=np.zeros((px[1],px[0]))
    imshow=cv2.ellipse(imshow,ellipse2,255,thickness=1)
    imshow=np.rot90(np.fliplr(imshow),1)
    plt.imshow(imshow)
    plt.scatter(data_zero_select[:,1],data_zero_select[:,0])
    plt.show()
    
    temp_loc_1=np.where(imshow>0)
    temp_2_max=np.max(temp_loc_1[0])
    temp_2_min=np.min(temp_loc_1[0])
    

    
    
    if len(temp[0])>3000:
        data_zero=data_zero[random.sample(list(range(len(temp[0]))),3000),:]
    else:
        data_zero=data_zero[random.sample(list(range(len(temp[0]))),max(700,int(len(temp[0])/5))),:]
    del temp



    data_min=np.min(data_zero,axis=0)
    data_max=np.max(data_zero,axis=0)
    data_min[0]=int((temp_1_min+temp_2_min)/2)
    data_max[0]=int((temp_1_max+temp_2_max)/2)
    center_point=np.mean(data_zero,axis=0)
    scale=(data_max-data_min)
    if scale[0]>0.6*(scale[1]+scale[2]):
        scale[0]=0.6*(scale[1]+scale[2])
        temp_1=int((center_point[0]-scale[0]/2))
        temp_2=int((center_point[0]+scale[0]/2))
    else:
        temp_1=data_min[0]
        temp_2=data_max[0]
        
        
    data_zero_add=np.array([[temp_1,center_point[1],center_point[2]],[temp_2,center_point[1],center_point[2]]])
    data_zero_bak=data_zero.astype(np.int).copy()
    data_zero=np.vstack([data_zero,data_zero_add])
    
    
    n=data_zero.shape[0]
    #bias_c=np.array((20,50,50))
    bias_c=((data_max-data_min)*0.4).astype(np.int)
    bias_c[0]=int(bias_c[0]*0.7) #空白区域留多少
    bias_xyz=bias_c-data_min
    bias_xyz=bias_xyz.astype(np.int)
    px=data_max+bias_xyz+bias_c+1
    px=px.astype(np.int)
    data_zero=data_zero+bias_xyz
    center_point=np.mean(data_zero,axis=0)
    center_point=center_point.astype(np.int)
    
    
    
    # n_in=1
    # data_in=np.zeros((1,3))
    # data_in[0,:]=center_point
    # data_in=data_in.astype(np.int)

    # n_in=10
    # index=random.sample(list(range(n)),n_in)
    # data_in=center_point+(0.2+0.4*np.random.rand(n_in,1))*(data_zero[index,:]-center_point)
    # data_in=data_in.astype(np.int)
    # data_in=del_repeat(data_in,px)
    # n_in=data_in.shape[0]
    
    
    n_in=10
    index=random.sample(list(range(n)),n_in)
    data_in=center_point+(0.2)*(data_zero[index,:]-center_point)
    data_in=data_in.astype(np.int)
    data_in=del_repeat(data_in,px)
    n_in=data_in.shape[0]
    
    
    n_out=int(min(300,np.sum((data==ind))/10)) # 不能太多
    data_out=np.zeros((n_out,3),dtype=np.int)
    for i in range(3):
        data_out[:,i]=np.random.randint(px[i],size=n_out)
    for i in range(n_out):
        chose=np.random.randint(6)
        data_out[i,chose//2]=0+chose%2*(px[chose//2]-1)
    
    
    
    
    
    
    el=(data_max-data_min)/2*1.5
#    el[0]=int(temp_0/2*1.5)
    k=1/np.sum((np.square(data_out-center_point)/np.square(el)),axis=1)
    k=np.minimum(np.sqrt(k),1)
    data_out=center_point+(data_out-center_point)*np.repeat(np.expand_dims(k,1),3,axis=1)
    data_out=data_out.astype(np.int)

    data_out=del_repeat(data_out,px)
    n_out=data_out.shape[0]
    
    
    data_range_max=np.max(data_out,axis=0)
    data_range_min=np.min(data_out,axis=0)
    
    
#    index=random.sample(list(range(n)),n_out)
#    data_out=data_zero[index,:]+(0.3+0.1*np.random.rand(n_out,1))*(data_zero[index,:]-center_point)
#    for i in range(3):
#        data_out[data_out[:,i]>=px[i],i]=px[i]-1
#    data_out[data_out<0]=0


#----------------------------------------------
    data_zero=data_zero.astype(np.int)
    mrc_out=np.zeros(px) 
    mrc_out[data_in[:,0],data_in[:,1],data_in[:,2]]=200
    mrc_out[data_out[:,0],data_out[:,1],data_out[:,2]]=50
    mrc_out[data_zero[:,0],data_zero[:,1],data_zero[:,2]]=120
    



#----------------------------------------------

    X=np.concatenate((data_zero,data_in,data_out),axis=0)
    X=X.astype(np.int)
    y=np.concatenate((np.repeat([0],n),np.repeat([-100.0],n_in),np.repeat([100.0],n_out)),axis=0)
    
    R=np.linalg.norm(px)
    print([R,px])
    
    sigma=1
    temp=np.zeros(px)
    temp2=np.where(temp==0)
    del temp
    X_all=np.zeros((len(temp2[0]),3))
    X_all[:,0]=temp2[0].flatten()
    X_all[:,1]=temp2[1].flatten()
    X_all[:,2]=temp2[2].flatten()
    del temp2
    Kx_x=Cov_mat(X,None,kernel_name,par)
    y_t=LA.solve((Kx_x+(sigma**2)*np.eye(Kx_x.shape[0],Kx_x.shape[1])),y)
    del Kx_x    
    
    r_mat=np.sqrt(np.square(0-np.transpose(np.expand_dims(X_all[:,0],1)))\
                 +np.square(0-np.transpose(np.expand_dims(X_all[:,1],1)))\
                 +np.square(0-np.transpose(np.expand_dims(X_all[:,2],1))))
    del X_all
    r_mat[r_mat==0]=1e-10
    
    A_mat=kernelfun(r_mat.T,kernel_name,par)
    del r_mat
    A_mat=np.squeeze(A_mat)
    
    y_index=X[:,0]*px[1]*px[2]+X[:,1]*px[2]+X[:,2]
    y_index_flag=np.zeros((px[0]*px[1]*px[2],),dtype=np.uint8)
    y_index_flag[y_index]=1
    
    y_pad=np.zeros((px[0]*px[1]*px[2],))
    y_pad[y_index]=y_t
        
    A_mat3=np.zeros((A_mat.shape[0]*8,))
    y_pad3=np.zeros((A_mat.shape[0]*8,))
    for i in range(px[0]):
        for j in range(px[1]):
            A_mat3[i*4*px[1]*px[2]+j*2*px[2]:i*4*px[1]*px[2]+j*2*px[2]+px[2]]=\
            A_mat[i*px[1]*px[2]+j*px[2]:i*px[1]*px[2]+j*px[2]+px[2]]
            A_mat3[i*4*px[1]*px[2]+j*2*px[2]+px[2]]=0
            A_mat3[i*4*px[1]*px[2]+j*2*px[2]+px[2]+1:i*4*px[1]*px[2]+j*2*px[2]+2*px[2]]=\
            A_mat[i*px[1]*px[2]+j*px[2]+px[2]-1:i*px[1]*px[2]+j*px[2]:-1]
            y_pad3[i*4*px[1]*px[2]+j*2*px[2]:i*4*px[1]*px[2]+j*2*px[2]+px[2]]=\
            y_pad[i*px[1]*px[2]+j*px[2]:i*px[1]*px[2]+j*px[2]+px[2]]
            y_pad3[i*4*px[1]*px[2]+j*2*px[2]+px[2]:i*4*px[1]*px[2]+j*2*px[2]+2*px[2]]=0
        A_mat3[i*4*px[1]*px[2]+2*px[1]*px[2]:i*4*px[1]*px[2]+2*px[1]*px[2]+2*px[2]]=0
        for j in range(1,px[1]):
            A_mat3[i*4*px[1]*px[2]+2*px[1]*px[2]+j*2*px[2]:i*4*px[1]*px[2]+2*px[1]*px[2]+j*2*px[2]+2*px[2]]=\
                A_mat3[i*4*px[1]*px[2]+(px[1]-j)*2*px[2]:i*4*px[1]*px[2]+(px[1]-j)*2*px[2]+2*px[2]]
        y_pad3[i*4*px[1]*px[2]+2*px[1]*px[2]:(i+1)*4*px[1]*px[2]]=0        
    
    del y_pad
    del A_mat        
    A_mat3[px[0]*(4*px[1]*px[2]):(px[0]+1)*(4*px[1]*px[2])]=0
    y_pad3[px[0]*(4*px[1]*px[2]):(px[0]+px[0])*(4*px[1]*px[2])]=0
    for i in range(1,px[0]):
        A_mat3[(px[0]+i)*(4*px[1]*px[2]):(px[0]+i+1)*(4*px[1]*px[2])]=\
        A_mat3[(px[0]-i)*(4*px[1]*px[2]):(px[0]-i+1)*(4*px[1]*px[2])]            
    
    f=np.fft.fftn(A_mat3.reshape(px[0]*2,px[1]*2,px[2]*2))
    del A_mat3
    g=np.fft.fftn(y_pad3.reshape(px[0]*2,px[1]*2,px[2]*2))
    del y_pad3
    z=f*g
    del f
    del g
    y3_0=np.real(np.fft.ifftn(z).flatten())
    del z
    #y0=np.dot(ans,y_pad)
    y3=np.zeros((px[0]*px[1]*px[2],))
    for i in range(px[0]):
        for j in range(px[1]):
            y3[i*px[1]*px[2]+j*px[2]:i*px[1]*px[2]+j*px[2]+px[2]]=\
            y3_0[i*px[1]*px[2]*4+j*px[2]*2:i*px[1]*px[2]*4+j*px[2]*2+px[2]]
    del y3_0
    mean=y3[y_index_flag==0] 
    del y3
    
    temp=np.zeros(px)
    for i in range(X.shape[0]):
        temp[X[i,0],X[i,1],X[i,2]]=1
    temp2=np.where(temp==0)
    del temp
    X_new=np.zeros((len(temp2[0]),3))
    X_new[:,0]=temp2[0].flatten()
    X_new[:,1]=temp2[1].flatten() 
    X_new[:,2]=temp2[2].flatten() 
    del temp2
    
    index_sample=random.sample(list(range(X_new.shape[0])),100)
    X_new_sample=X_new[index_sample,:]
    test_mat=Cov_mat(X_new_sample,X,kernel_name)    
    test_y=np.dot(test_mat,y_t)
    print(np.linalg.norm(test_y-mean[index_sample]))
    
    cc=300/np.max(px)
    index=np.where(y==0)[0]
    X=X-bias_xyz
    mean_mat[data_zero_bak[:,0],data_zero_bak[:,1],data_zero_bak[:,2]]=0  
    X_new=X_new.astype(np.int)    
    
    index=np.where(np.abs(mean)<=5)[0]
    
    mrc_out[X_new[index,0],X_new[index,1],X_new[index,2]]=20
    
    mean=np.abs(mean)
    index=np.where(np.abs(mean)<=cc)[0]
    X_new=X_new-bias_xyz
    index2=np.bitwise_and(np.bitwise_and(X_new[index,0]>=0 ,X_new[index,1]>=0),  X_new[index,2]>=0)
    for i in range(3):
        index2=np.bitwise_and(index2, X_new[index,i]<data.shape[i])
    index=index[index2==1]
    mean_mat[X_new[index,0],X_new[index,1],X_new[index,2]]=(mean_mat[X_new[index,0],X_new[index,1],X_new[index,2]]>mean[index])*mean[index]+\
        (mean_mat[X_new[index,0],X_new[index,1],X_new[index,2]]<=mean[index])*mean_mat[X_new[index,0],X_new[index,1],X_new[index,2]]
    

    image_path=str(ind)+'mean_point.mrc'
    if os.path.exists(image_path):
        mrc2=mrcfile.open(image_path,mode='r+')
    else:
        mrc2=mrcfile.new(image_path)
    mrc2.set_data(mrc_out.astype(np.uint8))
    mrc2.close()    

save_list=['mean_mat']
import pickle as pk
for par in save_list:
    f = open(par,'wb')
    pk.dump(locals()[par],f)
    f.close()

#mrc_out=np.zeros(mean_mat.shape) 
#index=np.bitwise_and(np.bitwise_and(X[:,0]>=0 ,X[:,1]>=0),  X[:,2]>=0)
#for i in range(3):
#    index=np.bitwise_and(index, X[:,i]<data.shape[i])
#index=np.bitwise_and(index, y[:]==0)
#mrc_out[X[index,0],X[index,1],X[index,2]]=50+(y[index]//100+1)*50
#
#image_path='mean_point_0.mrc'
#if os.path.exists(image_path):
#    mrc2=mrcfile.open(image_path,mode='r+')
#else:
#    mrc2=mrcfile.new(image_path)
#mrc2.set_data(mrc_out.astype(np.uint8))
#mrc2.close()
# 
#mrc_out=np.zeros(mean_mat.shape) 
#index=np.bitwise_and(np.bitwise_and(X[:,0]>=0 ,X[:,1]>=0),  X[:,2]>=0)
#for i in range(3):
#    index=np.bitwise_and(index, X[:,i]<data.shape[i])
#index=np.bitwise_and(index, y[:]==100)
#mrc_out[X[index,0],X[index,1],X[index,2]]=50+(y[index]//100+1)*50
#
#image_path='mean_point_out.mrc'
#if os.path.exists(image_path):
#    mrc2=mrcfile.open(image_path,mode='r+')
#else:
#    mrc2=mrcfile.new(image_path)
#mrc2.set_data(mrc_out.astype(np.uint8))
#mrc2.close()
##  
cc=3
mrc_out=np.zeros(mean_mat.shape) 
temp=np.where(np.abs(mean_mat)<=cc)
mrc_out[temp[0],temp[1],temp[2]]=200
temp=np.where((np.abs(mean_mat)==0))
mrc_out[temp[0],temp[1],temp[2]]-=100

image_path='mean_surface_debuge.mrc'
if os.path.exists(image_path):
    mrc2=mrcfile.open(image_path,mode='r+')
else:
    mrc2=mrcfile.new(image_path)
mrc2.set_data(mrc_out.astype(np.uint8))
mrc2.close()



