# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import sys
import math
import random
import cv2
from matplotlib import pyplot as plt
from math import *
import time
import glob
import os
import mrcfile
from numpy.linalg import norm

ACTION = ["N", "S", "E", "W","NE", "SE", "SW", "NW"]
COMPASS = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
    "NE": (1, -1),        
    "SE": (1, 1),
    "SW": (-1, 1),
    "NW": (-1, -1)
}
    
def good_action(action,action_vec):

    dir=COMPASS[ACTION[action]]
    dir_0=action_vec 
    if dir[0]*dir_0[0]+dir[1]*dir_0[1]>=0:
        return(True)
    else:
        return(False)
        
def inner_product(a,b):
    ans=(a[0]*b[0]+a[1]*b[1])/(np.linalg.norm(a)*np.linalg.norm(b))
    if isnan(ans):
        ans=0
    return(ans)  
   
def cal_rot(x,theata):
    trans_mat=np.array([(cos(theata),-sin(theata)),(sin(theata),cos(theata))])
    return(np.dot(trans_mat,x))
    
def cal_angle(x,y):
   return(acos(max(min(1,np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)),0)))     
    
def cal_dis_withk(x2,x1,k):
    kk=k[1]/k[0]
    dis=math.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)
    dis+=abs(x2[1]-kk*(x2[0]-x1[0])-x1[1])/math.sqrt(kk**2+1)
    return(dis)

def cal_dis(x2,x1):
  
    dis=math.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)
    
    return(dis)

def dis(a,b):
    if isinstance(a,list):
        a=np.array(a)
    if isinstance(b,list):
        b=np.array(b)
        
    return(np.linalg.norm(b-a))

def cal_scale(x):
    return(dis(np.array([np.min(x[:,0]),np.min(x[:,1])]),np.array([np.max(x[:,0]),np.max(x[:,1])])))
    
def tranxy(ind,m):
  q=len(ind)
  cor_xy=np.zeros((q,2))
  cor_xy[:,0]=(ind)//m
  cor_xy[:,1]=ind%m
  cor_xy=cor_xy.astype(np.int)
  return(cor_xy)
  
def cal_dir(x,type):
    n_chose=10
    if type=='end':
        work=x
    else:
        work=x[::-1]
            
    if len(x)>n_chose:
        work=work[-n_chose:]
    else:
        work=work
        
    dir_ans=np.array([0.0,0.0])
    for i in range(1,len(work)):
        dir_ans+=0.1*((work[i][0]-work[i-1][0],work[i][1]-work[i-1][1])-dir_ans)
        dir_ans=dir_ans/np.linalg.norm(dir_ans) 
    
    if type!='end':
        dir_ans=-dir_ans
    
    return(dir_ans)

def cal_jump_dis(p_s,dir_s,state_temp,l_s):

    
    p_e=state_temp[0]
    dir_e=cal_dir(state_temp,'start')
    l_e=len(state_temp)
    dir_m=np.array((p_e[0]-p_s[0],p_e[1]-p_s[1]))
    if np.linalg.norm(dir_m)>0:
        dir_m=dir_m/np.linalg.norm(dir_m)
    
            
    
    if inner_product(dir_m-dir_s,dir_e-dir_m)<-0.1 and cal_dis(p_s,p_e)>=30:
        return((1000,1000,1000))
    else:
        #print(cal_dis(dir_s,dir_m))
#        print(math.acos(inner_product(dir_s,dir_m))/math.pi*180)
#
#        #print(cal_dis(dir_m,dir_e))
#        print(math.acos(inner_product(dir_m,dir_e))/math.pi*180)
#        print(cal_dis(p_s,p_e))
#        print('------------')
        val=(l_s*cal_dis(dir_s,dir_m)+l_e*cal_dis(dir_m,dir_e))/(l_s+l_e)
        
        #return(cal_dis(dir_s,dir_m)+cal_dis(dir_m,dir_e)+cal_dis(p_s,p_e)/len(state_temp))
        return (val,val,cal_dis(p_s,p_e))
def cal_jump_cost(dir_s,p_s,dir_e,p_e):

    pinf=-10000

    dir_m=np.array((p_e[0]-p_s[0],p_e[1]-p_s[1]))
    if np.linalg.norm(dir_m)>0:
        dir_m=dir_m/np.linalg.norm(dir_m)
    
            
    
    if (inner_product(dir_m,dir_s)<0 or inner_product(dir_e,dir_m)<0) and cal_dis(p_s,p_e)>=30:
        return(pinf)
    else:
        #print(cal_dis(dir_s,dir_m))
        ag1=math.acos(max(min(1,inner_product(dir_s,dir_m)),-1))/math.pi*180
        ag2=math.acos(max(min(1,inner_product(dir_m,dir_e)),-1))/math.pi*180
#        print(cal_dis(p_s,p_e))
#        print('------------')
        val=-(ag1/180+ag2/180+cal_dis(p_s,p_e)/100)
        
        #return(cal_dis(dir_s,dir_m)+cal_dis(dir_m,dir_e)+cal_dis(p_s,p_e)/len(state_temp))
        return (val)


def AbetterthanB(A,B):
    if A[0]/B[0]*A[1]/B[1]*A[2]/B[2]<1:
        return(True)
    else:
        return(False)   

def scatterlist(x_list):
    plt.figure()
    for i,x in enumerate(x_list):
        x=np.array(x)
        plt.scatter(x[:,0],x[:,1],label=str(i),zorder=1)
#        plt.scatter(x[0,0],x[0,1],1,label=str(i)+'s',zorder=2)
#        plt.scatter(x[-1,0],x[-1,1],1,label=str(i)+'e',zorder=2)
    plt.legend()
    plt.xlim([0,512])
    plt.ylim([0,512])
    #plt.savefig('temp.jpg',dpi=500)
    
    #plt.show() 
    
def scatterindex(x_list,state_all):
    plt.figure()
    st=[-1,-1]
    for i,xx in enumerate(x_list):
        temp=state_all[xx//2]
        if xx%2==1:
            temp=temp[::-1]
        x=np.array(temp)    
        plt.scatter(x[:,0],x[:,1],label=str(i),zorder=1,c='b')
        if i==0:
            plt.scatter(x[0,0],x[0,1],1,label=str(i)+'s',zorder=2,c='r')
        if i==len(x_list)-1:    
            plt.scatter(x[-1,0],x[-1,1],1,label=str(i)+'e',zorder=2,c='y')
        if i>0:
            plt.plot([st[0],temp[0][0]],[st[1],temp[0][1]],c='k')
        st=temp[-1]    
            
    #plt.legend()
    plt.xlim([0,512])
    plt.ylim([0,512])
#    plt.savefig('temp.jpg',dpi=500)
#    
#    plt.show()    

def fitandpridect(state_list,iffig):

    if len(state_list)<6:
        return((-inf,-1),(-1,-1),-1)
    ellipse = cv2.fitEllipse(state_list.astype(np.int))
    if iffig:
        plt.figure()
        imshow=np.zeros((512,512))
        img_res=cv2.ellipse(imshow,ellipse,255,thickness=2)
        plt.imshow(img_res)
        plt.scatter(state_list[:,0],state_list[:,1],1)
        plt.show()

#     
#  
#        plt.savefig('temp.jpg',dpi=500)
    return(ellipse) 

def cal_ellipse_dis(ellipse,state_to):   
    if ellipse[0][0]==-inf:
        return(0)
    def cal_rot(x,theata):
        trans_mat=np.array([(math.cos(theata),-math.sin(theata)),(math.sin(theata),math.cos(theata))])
        return(np.dot(trans_mat,x.T)) 
    center=ellipse[0]
    axisl=ellipse[1]
    angle=ellipse[2]
    
    state_to_rot=cal_rot(np.array(state_to)-np.array(center),-angle/180*math.pi)
    s= np.square(state_to_rot[0,:])/((axisl[0]/2)**2)+np.square(state_to_rot[1,:])/((axisl[1]/2)**2)
    
    #s=(state_to_rot[0]**2)/((axisl[0]/2)**2)+(state_to_rot[1]**2)/((axisl[1]/2)**2)
    return(np.abs(1-1/np.sqrt(s)))

def good_ellipse(el1,el2,el3):
    if np.linalg.norm(el1[0:2]-el3[0:2])>20 and  np.linalg.norm(el2[0:2]-el3[0:2])>20:
        return(False)
    elif np.linalg.norm(el1[2:4]-el3[2:4])/(np.linalg.norm(el1[2:4])+np.linalg.norm(el3[2:4])) >0.5 \
        and np.linalg.norm(el2[2:4]-el3[2:4])/(np.linalg.norm(el2[2:4])+np.linalg.norm(el3[2:4])) >0.5  :
        return(False)
    else:
        return(True)

def good_ellipse2(el3,state_new,m):
    s=cal_ellipse_dis(el3,state_new)
    #print(np.max(s))
    if np.max(s)>0.1 and (np.mean(s[:m])>0.05 or np.mean(s[m:])>0.05) :
        return(False)
    else:
        return(True)      

def same_ellipse(el1,el2):
    el1_0=np.array(el1[0])
    el2_0=np.array(el2[0])
    el1_1=np.array(el1[1])
    el2_1=np.array(el2[1])

    if norm(el1_0-el2_0)/(0.5*(norm(el1_0)+norm(el2_0)))<0.05 and \
       norm(el1_1-el2_1)/(0.5*(norm(el1_1)+norm(el2_1)))<0.3:
           return(True)
    else:
        return(False)

def select_action(state, explore_rate,action_0):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
        while not(good_action(action,action_0)):
            action = env.action_space.sample()    
    # Select the action with the highest q
    else:
        best=-100
        action_1=-1
        for action in range(q_table.shape[2]):
            if (q_table[state+(action,)]>best) and good_action(action_0,action):
                action_1 = action
                best=q_table[state+(action,)]
            elif  (q_table[state+(action,)]==best) and good_action(action_0,action):
                action_1 =random.sample([action_1,action],1)[0]
                
        action=action_1        
    return action





    
