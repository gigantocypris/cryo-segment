#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:08:28 2019

@author: lizhou
"""
import matplotlib as mpl
# mpl.use('Agg')

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
import pickle as pk
from scipy import interpolate 
from toolbox import *


filename='mon_t1_trimmed_rec_nadunet_membrane.mrc'
workid=7
iffig=False
iffigv=False

if not os.path.exists('results-'+str(workid)):
    os.mkdir('results-'+str(workid))
if not os.path.exists('data-'+str(workid)):
    os.mkdir('data-'+str(workid))
    
   
np.random.seed ( 30 )
random.seed(139)
# px=512
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
   
foe=5
MAX_T=10000


data_path="../../2.U-net-new/Pytorch-UNet-master/"
filename=data_path+'mon_t2_trimmed_rec_nadunet_membrane.mrc'

data_path="../../3.Union-find/"
datafilename=data_path+'mon_t2_trimmed_rec_nadunet_membrane_mrc_2D_150_clean.mrc'


image_path=filename.replace('membrane','image')
mrc1=mrcfile.open(image_path)
data_img=mrc1.data

#maze=cv2.imread('mon_t1_trimmed_rec_nad_100_out_membrane.png',cv2.IMREAD_COLOR).astype(np.uint8)



image_path=datafilename
mrc1=mrcfile.open(image_path)
data1=mrc1.data
px=data1.shape[1:]



f = open('data-'+str(workid)+'/ellipse_ans_list','rb')
ellipse_ans_list=pk.load(f)
f.close()
   
    

s_ind=1000
e_ind=0
for temp in ellipse_ans_list:
    if temp[0]<s_ind:
        s_ind=temp[0]
    if temp[0]>e_ind:
        e_ind=temp[0]
        
ellipse_order=[]
n=e_ind-s_ind+1
for i in range(n):
    ellipse_order.append([])
    
for temp in ellipse_ans_list:
    ellipse_order[temp[0]-s_ind]=temp[1]

ellipse_list=[]
ellipse_order_index=[]
for q,ellipse_list_temp in enumerate(ellipse_order):
    ellipse_order_index.append([])
    for p,ellipse in enumerate(ellipse_list_temp):
        
        best_i=-1
        if len(ellipse_list)>0:
            for i,ellipse_c in enumerate(ellipse_list):
                if same_ellipse2(ellipse_c,ellipse):
                     best_i=i
                     ellipse_list[i]=ellipse 
                     break
        if best_i==-1:
            ellipse_list.append(ellipse)
            best_i=len(ellipse_list)-1
        ellipse_order_index[q].append(best_i)

        
m=len(ellipse_list)
ellipse_cal=np.zeros((m,3))
ellipse_cal[:,0]=1000
for p,temp in enumerate(ellipse_order_index):
    for j in range(m):
        if j in temp:
            if p<ellipse_cal[j,0]:
                ellipse_cal[j,0]=p
            if p>ellipse_cal[j,1]:
                ellipse_cal[j,1]=p
            ellipse_cal[j,2]+=1


th=4
#椭圆检查
for p,p_ellipse in enumerate(ellipse_list):
    temp=p_ellipse[1]    
    if (max(temp)/(min(temp)+1e-10)>2.1):
        ellipse_cal[p,2]=0
        
for p,p_ellipse in enumerate(ellipse_list):
    for q in range(p+1,len(ellipse_list)):
        q_ellipse=ellipse_list[q]
        #if (np.linalg.norm([p_ellipse[0][0]-q_ellipse[0][0],p_ellipse[0][1]-q_ellipse[0][1]])<1.2*(max(p_ellipse[1][0],q_ellipse[1][0])+max(p_ellipse[1][1],q_ellipse[1][1]))):
        if same_ellipse3(p_ellipse, q_ellipse):
            if ellipse_cal[p,2]>=ellipse_cal[q,2]:
                ellipse_cal[q,2]=0
            else:
                ellipse_cal[p,2]=0

for p,p_ellipse in enumerate(ellipse_list):
    for q in range(p+1,len(ellipse_list)):
        q_ellipse=ellipse_list[q]
        
        if ellipse_cal[p,2]>th and ellipse_cal[q,2]>th:
            check=np.zeros((px[1],px[0]))
            imshow=np.zeros((px[1],px[0]))
            imshow=cv2.ellipse(imshow,p_ellipse,100,thickness=1)
            check=check+imshow
            imshow=np.zeros((px[1],px[0]))
            imshow=cv2.ellipse(imshow,q_ellipse,100,thickness=1)
            check=check+imshow
            if np.sum(check==200)>5:
                print([p,q])
                if ellipse_cal[p,2]>=ellipse_cal[q,2]:
                    ellipse_cal[q,2]=0
                else:
                    ellipse_cal[p,2]=0
            
ellipse_cal[45,2]=0               
         
ellipse_ans_list=[]
for p,temp in enumerate(ellipse_order):
    
    ellipse_ans_list.append([])
    for q in range(m):
        if p>=ellipse_cal[q,0] and p<=ellipse_cal[q,1] and ellipse_cal[q,2]>th:
            if q in ellipse_order_index[p]:
                ellipse_ans_list[p].append(temp[ellipse_order_index[p].index(q)])
            else:
                ellipse_ans_list[p].append([])
        else:
             ellipse_ans_list[p].append([])
#             

kind_n=0
kind_list={}
for q in range(m):
    temp=ellipse_list[q][1]
    if  ellipse_cal[q,2]>th and (max(temp)/min(temp)<2.1):
        kind_n=kind_n+1
        kind_list[q]=kind_n
        for p in range(int(ellipse_cal[q,0]),int(ellipse_cal[q,1])):
            if ellipse_ans_list[p][q]==[]:
                p1=p-1
                while ellipse_ans_list[p1][q]==[]:
                    p1=p1-1
                p2=p+1            
                while ellipse_ans_list[p2][q]==[]:
                    p2=p2+1
                ellipse_ans_list[p][q]=((0.5*(ellipse_ans_list[p1][q][0][0] \
                                            +ellipse_ans_list[p2][q][0][0]),
                                         0.5*(ellipse_ans_list[p1][q][0][1] \
                                            +ellipse_ans_list[p2][q][0][1])),
                                        (0.5*(ellipse_ans_list[p1][q][1][0] \
                                            +ellipse_ans_list[p2][q][1][0]),
                                         0.5*(ellipse_ans_list[p1][q][1][1] \
                                            +ellipse_ans_list[p2][q][1][1])),
                                         0.5*(ellipse_ans_list[p1][q][2] \
                                            +ellipse_ans_list[p2][q][2]))
        
        p1=int(ellipse_cal[q,0])
        for p in range(p1-1,-1,-1):
            ellipse_ans_list[p][q]=ellipse_ans_list[p1][q]
        p1=int(ellipse_cal[q,1])

        for p in range(p1+1,len(ellipse_ans_list)):
            ellipse_ans_list[p][q]=ellipse_ans_list[p1][q]
        

                       

val_foe=20
val_mat=np.zeros((2*val_foe+1,2*val_foe+1))
for x in range(val_foe*2+1):
    for y in range(val_foe*2+1):
        val_mat[x,y] = math.sqrt((x-val_foe)**2+(y-val_foe)**2)



#indd_list=list(range(120,224+1))
#indd_list.extend(list(range(119,0-1,-1)))
indd_list=list(range(120,134))
indd_list.extend(list(range(119,70,-1)))
#indd_list=list(range(120,130))
#indd_list.extend(list(range(119,110,-1)))
#indd_list=[86]

imgoutput=np.zeros((data1.shape[0],px[0],px[1]))
imgoutput_dilate=imgoutput.copy()



for indd in indd_list:  
    
    print(['epoch',indd])
    maze=np.squeeze(data1[indd,:,:])
    maze=maze//255

    state_list=np.zeros((MAX_T,3))
    state_index=0
   
    plt.figure(1)
    plt.ion()
    mazedata=maze
    foe_mat=np.zeros((2*foe+1,2*foe+1))
    for x in range(2*foe+1):
        for y in range(2*foe+1):
            foe_mat[x,y]=1/(abs(x-foe)**1+abs(y-foe)**1+1e-10)
    foe_mat[foe,foe]=2        
    mazevalue=np.zeros(mazedata.shape)
    for x in range(mazedata.shape[0]):
        for y in range(mazedata.shape[1]):
              robot=np.array([x,y])
              xx=range(max(x-foe,0),min(x+foe+1,mazedata.shape[0]))
              yy=range(max(y-foe,0),min(y+foe+1,mazedata.shape[1]))
              mazevalue[x,y]=np.sum(mazedata[xx,:][:,yy]*foe_mat[xx-robot[0]+foe,:][:,yy-robot[1]+foe])
              
    
    
    save_list=['state_all']
    for par in save_list:
        f = open('data-'+str(workid)+'/'+str(indd)+'_'+par,'rb')
        locals()[par]=pk.load(f)
        f.close()
    ellipse_ans=ellipse_ans_list[indd-s_ind].copy() 
    
    ellipse_ans_kind=[]
    for i,ellipse in enumerate(ellipse_ans):
        if ellipse !=[]:
            ellipse_ans_kind.append(kind_list[i])
    i=0
    while i<len(ellipse_ans):
        if ellipse_ans[i]==[]:
            del(ellipse_ans[i])
        else:

            i=i+1
            
    
    i=0
    while i<len(state_all):
        if len(state_all[i])<6:
            del(state_all[i])
        else:
            i=i+1
    
    
    state_list=np.zeros((MAX_T,3))
    
    state_index=1
    state_i=0
    done=False
    
    n=len(state_all)
    notused=np.zeros([n,1])+1
    len_list=np.zeros([n,1])
    ellipse_list=np.zeros([n,4])
    
    
    for i,state_now in enumerate(state_all):
        len_list[i]=len(state_now)
        ellipse_temp=fitandpridect(np.array(state_now),iffig)
        ellipse_list[i,0:2]=np.array(ellipse_temp[0])
        ellipse_list[i,2:4]=np.array(ellipse_temp[1])
        
    if len(ellipse_ans)>0:
        diss=np.zeros([len(state_all),len(ellipse_ans),2])
        for i,state_now in enumerate(state_all):    
            for j,ellipse_temp in enumerate(ellipse_ans):
                s=cal_ellipse_dis(ellipse_temp,state_now)
                diss[i,j,0]=np.max(s)
                diss[i,j,1]=np.mean(s)
        
        
        state_pre=[]
        for j,ellipse_temp in enumerate(ellipse_ans):
            state_now=[]
            state_now_index=[]
            
            
            
            for i,state_temp in enumerate(state_all):    
                s=cal_ellipse_dis(ellipse_temp,state_temp)
                if np.max(s)<0.25 and np.mean(s)<0.1 and notused[i]==1:
                    state_now.append(state_temp)
                    state_now_index.append(i)
                    notused[i]=0
                    if len(state_now)==1:
                        state_now_np=np.array(state_temp)
                    else:
                        state_now_np=np.concatenate((state_now_np,np.array(state_temp)),axis=0)
            
            
            for p1 in range(len(state_now)-1):
                for p2 in range(p1+1,len(state_now)):
                    if len(state_now[p1])<len(state_now[p2]):
                        state_temp=state_now[p1]
                        state_now[p1]=state_now[p2]
                        state_now[p2]=state_temp
                        state_temp=state_now_index[p1]
                        state_now_index[p1]=state_now_index[p2]
                        state_now_index[p2]=state_temp
                        
            if len(state_now)>0:
                for p1 in range(1,len(state_now)):
                    best_p_val=10000
                    best_p_state=[]
                    best_p_index=0
    
                    p_s=state_now[p1-1][-1]
                    dir_s=cal_dir(state_now[p1-1],'end')
                    
                    
                    temp_val=cal_jump_dis(p_s,dir_s,state_now[0],len(state_now[p1-1]))
                    best_p_val=temp_val[2]
                    best_p_index=0
                    
                    
                    for p2 in range(p1,len(state_now)):
                        state_temp=state_now[p2]
                        temp_val=cal_jump_dis(p_s,dir_s,state_temp,len(state_now[p1-1]))
                        if temp_val[2]<best_p_val:
                            best_p_val= cal_dis(state_now[p1-1][-1],state_now[p2][0])
                            best_p_state=state_now[p2]
                            best_p_index=p2
                        state_temp=state_now[p2][::-1]
                        temp_val=cal_jump_dis(p_s,dir_s,state_temp,len(state_now[p1-1]))
                        if temp_val[2]<best_p_val:
                            best_p_val= cal_dis(state_now[p1-1][-1],state_now[p2][-1])
                            best_p_state=state_now[p2][::-1]
                            best_p_index=p2
                    if best_p_index!=0:        
                        state_now[best_p_index]=state_now[p1]
                        state_now[p1]=best_p_state
                    else:
                        for p2 in range(p1,len(state_now)):
                            notused[state_now_index[p2]]=1
                            state_now[p2]=[]
                            state_now_index[p2]=[]
                        break
            p=0
            while p<len(state_now):
                if state_now[p]==[]:
                    del(state_now[p])
                else:
                    p=p+1
                    
                    
            if len(state_now)>0:        
                state_pre.append(state_now)            
    
    
        for state_temp in state_pre:
            for best_state in state_temp:
                state_list[state_i:state_i+len(best_state),0:2]=np.array(best_state)
                state_list[state_i:state_i+len(best_state),2]=state_index
                state_i=state_i+len(best_state)
            state_list[state_i,0:2]=np.array(state_temp[0][0])
            state_list[state_i,2]=state_index
            state_i=state_i+1    
            state_index=state_index+1
        state_list=state_list[:state_i,:]
    
################################################################################################
#Part 6:                    
################################################################################################
    iffigv=False
    maze_img=np.squeeze(data_img[indd,:,:])
    
    
    state_list=state_list.astype(np.int)  
    state_ans=[]
    learningrate=0.2
    if len(state_list)>0:
        for ind in range(1,1+int(np.max(state_list[:,2]))):
        #for ind in [2]:
        
            ind_list=state_list[:,2]==ind
            if np.sum(ind_list)<6:
                continue
            state_list_now=state_list[ind_list,0:2]
            if (np.linalg.norm(state_list_now[0,:]-state_list_now[-1,:])>0) and \
                (np.linalg.norm(state_list_now[0,:]-state_list_now[-1,:])/np.sum(ind_list)<0.15):
                state_list_now= np.row_stack((state_list_now,state_list_now[0,:]))
               
            if state_list_now[0,0]!=state_list_now[-1,0] or state_list_now[0,1]!=state_list_now[-1,1]:
                continue
                
            state_list_now=np.row_stack((state_list_now,np.array([[1000,1000]])))
            state_new=[[state_list_now[0,0],state_list_now[0,1]]]
            state_new_list=[]
            state=state_list_now[0,:]
            
            p=0
            target=[state_list_now[state_list_now.shape[0]-2,0],state_list_now[state_list_now.shape[0]-2,1]]
            action_vec=np.array([state_list_now[p+1,0]-state_list_now[p,0],state_list_now[p+1,1]-state_list_now[p,1]])
            action_vec=action_vec/np.linalg.norm(action_vec)    
            
            while p < len(state_list_now):
                #print(state)
                best_state=np.array([-1,-1])
                best_value=-1
                retarget=False
                for i in range(8) :
                    dir=COMPASS[ACTION[i]]
                    state_1=state+dir
                    if inner_product(dir,(state_list_now[p+1,0]-state_list_now[p,0],state_list_now[p+1,1]-state_list_now[p,1]))>0.5:
                        if mazevalue[state_1[0],state_1[1]]>best_value and mazedata[state_1[0],state_1[1]]>0 :
                            best_state=state_1
                            best_value=mazevalue[state_1[0],state_1[1]]
                if best_value<=0 :
                    
                    while (p<state_list_now.shape[0]-3)  and (dis(state_list_now[p+1,:],state_list_now[p,:])<1.5 \
                     or dis(state,state_list_now[p,:])>=dis(state,state_list_now[p+1,:])):
                        p=p+1
                        
                best_state=state_list_now[p+1,:]
                action_vec+=learningrate*(np.array(best_state).astype(np.float)-np.array(state).astype(np.float)-action_vec)
                action_vec=action_vec/np.linalg.norm(action_vec)                
                state=best_state
                state_new.append([state[0],state[1]])  
                    
                if (not(retarget) and state[0]==target[0] and state[1]==target[1]) :
                    break  
                while (p<state_list_now.shape[0]-3) and dis(state,state_list_now[p,:])>=dis(state,state_list_now[p+1,:]):
                    p=p+1 
                    
                     
            if len(state_new)>2:
                state_new_list.append(state_new)
                
            #print (ind)
        
            for ind2,state_new in enumerate(state_new_list):
                state_final=[]
                for i in range(len(state_new)):
                    if mazedata[state_new[i][0],state_new[i][1]]==1:
                         state_final.append(state_new[i])       
                if len(state_final)>0:
                    state_new=state_final        
                    state_new=np.asarray(state_new)  
                    if iffigv:
                        plt.figure()
                        plt.imshow(np.rot90(np.fliplr(mazedata),1))
                        plt.plot(state_new[:,0],state_new[:,1])
            #            plt.scatter(state_new[:,0],state_new[:,1],10,color='',marker='o',edgecolors='g',)
                        plt.scatter(state_new[:,0],state_new[:,1],1)
                        plt.title('results-'+str(workid)+'/'+str(ind)+'_'+str(dis(state_new[0,:],state_new[-1,:]) ))
                        plt.savefig('results-'+str(workid)+'/'+str(indd)+'_step6_'+str(ind)+'_'+str(ind2)+'.png',dpi=100)
                        #plt.show() 
                    
                    #if dis(state_new[0,:],state_new[-1,:])<0.2*cal_scale(state_new) and (state_new.shape[0]>10):
                    #    state_ans.append(state_new)
                    if  (state_new.shape[0]>30):
                        state_ans.append(state_new)
    robot=[0,0]
    search_range=5
    n=10
    mazedis=np.zeros(mazedata.shape)+10000
    
 
    # 此处开始依据椭圆，融合修复    
    state_ans_fix=[]
    state_ans_fix_kind=[]
    for ind,state_new in enumerate(state_ans):  
        best_ellipse=-1
        best_q=10000
        best_kind=-1
        for temp_ind,ellipse in enumerate(ellipse_ans):
            temp_q=np.max(cal_ellipse_dis(ellipse,state_new))
            if temp_q < best_q:
                best_ellipse=ellipse
                best_q=temp_q
                best_kind=ellipse_ans_kind[temp_ind]
        state_ans_fix_kind.append(best_kind)
        imshow=np.zeros((px[1],px[0]))
        imshow=cv2.ellipse(imshow,best_ellipse,255,thickness=1)
        imshow=np.rot90(np.fliplr(imshow),1)

        
        best_state=[-10,10]
        best_q=1000
    
        temp=np.where(imshow>0)
        for i in range(temp[0].shape[0]):
            if (state_new[0][0]-temp[0][i])**2+(state_new[0][1]-temp[1][i])**2<best_q:
                best_q=(state_new[0][0]-temp[0][i])**2+(state_new[0][1]-temp[1][i])**2
                best_state=(temp[0][i],temp[1][i])
        best_state_1=best_state
        
        
        
       
        best_q=1000
        dir_s=cal_dir_avg_start(state_new,'start',20)
        for q in range(8) :
                dir=COMPASS[ACTION[q]]
                state_1=np.array(best_state_1)+dir
                if imshow[state_1[0],state_1[1]]>100 \
                    and inner_product(dir_s,dir)>=0     \
                and (state_new[1][0]-state_1[0])**2+(state_new[1][1]-state_1[1])**2<best_q:
                    best_q=(state_new[1][0]-state_1[0])**2+(state_new[1][1]-state_1[1])**2
                    best_state_2=(state_1[0],state_1[1])    
        
                
        state_ref=[]
        state_ref.append(best_state_1)
        state_ref.append(best_state_2)
        state=best_state_2
        imshow[best_state_1[0],best_state_1[1]]=1
        imshow[best_state_2[0],best_state_2[1]]=1
        imshow[best_state_1[0],best_state_2[1]]=1
        imshow[best_state_2[0],best_state_1[1]]=1
        
        for i in range(2,temp[0].shape[0]):
            for q in range(8) :
                dir=COMPASS[ACTION[q]]
                state_1=np.array(state)+dir
                if imshow[state_1[0],state_1[1]]>100 and \
               ((i>4) or inner_product(np.array(state_ref[-1])-np.array(state_ref[-2]),state_1-np.array(state_ref[-1]))>=0):
                    state=state_1
                    imshow[state_1[0],state_1[1]]=1
                    state_ref.append((state[0],state[1]))
                    break
        state_ref.append((best_state_1[0],best_state_1[1]))        
        state_new_fix=[(state_new[0,0],state_new[0,1])]
        p=0
        q=0
        while cal_dis(state_ref[q+1],state_new[p])<cal_dis(state_ref[q],state_new[p]):
            q=q+1
        while p<len(state_new)-1:
        
                    
            if cal_dis(state_new[p],state_new[p+1])>1.9:
                if cal_dis(state_ref[q],state_new[p])<0.5:
                    imshow[state_ref[q]]=0
                    q=q+1
                    
                x = state_new[p+1][0]
                y = state_new[p+1][1]
                xx=range(max(x-val_foe,0),min(x+val_foe+1,imshow.shape[0]))
                yy=range(max(y-val_foe,0),min(y+val_foe+1,imshow.shape[1]))
                temp=imshow[xx,:][:,yy]*val_mat[xx-x+val_foe,:][:,yy-y+val_foe]
                temp[temp==0]=100000
                temp_ind=np.argmin(temp)
                state_target=(xx[temp_ind//len(yy)],yy[temp_ind%len(yy)])
                

                
                while q<len(state_ref)-1 and (cal_dis(state_ref[q+1],state_target)>1.5): 

                    state_new_fix.append(state_ref[q])
                    imshow[state_ref[q]]=0
                    q=q+1

                    
                    
#______________________________________________________________________________                    
            if  cal_dis(state_new_fix[-1],state_new[p+1])>0:  
                state_new_fix.append((state_new[p+1,0],state_new[p+1,1]))
 
                
            p=p+1
            while (q<len(state_ref)-1 and cal_dis(state_ref[q+1],state_new[p])<=cal_dis(state_ref[q],state_new[p])) or \
                  (q<len(state_ref)-2 and cal_dis(state_ref[q+2],state_new[p])<=cal_dis(state_ref[q],state_new[p])) :
                imshow[state_ref[q]]=0
                q=q+1
                
        #此处修复断点
        p=0
        while p<len(state_new_fix)-1:
            if cal_dis(state_new_fix[p],state_new_fix[p+1])<2:
                p=p+1
            else:
                state_1=np.array(state_new_fix[p])
                state_2=np.array(state_new_fix[p+1])
                scale=math.ceil(cal_dis(state_1,state_2))
                temp_list=[state_1]
                for i in range(scale-1):
                    temp=np.round(state_1+(i+1)/scale*(state_2-state_1)).astype(np.int32)
                    if cal_dis(temp,state_2)==0:
                        break
                    if cal_dis(temp,temp_list[-1])>0:
                        temp_list.append(temp)
                for i,temp in enumerate(temp_list[1:]):
                    state_new_fix.insert(p+i+1,(temp[0],temp[1]))
                    
                
        state_ans_fix.append(np.array(state_new_fix))    

    
    plt.figure() 
    plt.imshow(np.rot90(np.fliplr(maze_img),1),cmap ='gray')
    for ind,state_new in enumerate(state_ans_fix):
        plt.scatter(state_new[:,0],state_new[:,1],2,'b',zorder=3)
        imgoutput[indd,state_new[:,0],state_new[:,1]]=state_ans_fix_kind[ind]


    
    plt.axis('off')
    plt.savefig('results-'+str(workid)+'/'+str(indd)+'_step9_fix'+'.png',dpi=100)



if os.path.exists('img_fix_show'+'_'+str(workid)+'.mrc'):
    mrc2=mrcfile.open('img_fix_show'+'_'+str(workid)+'.mrc',mode='r+')
else:
    mrc2=mrcfile.new('img_fix_show'+'_'+str(workid)+'.mrc')
mrc2.set_data(imgoutput.astype(np.uint8))
mrc2.close()

for indd in indd_list:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
    imgoutput_dilate[indd,:,:] = cv2.dilate(np.squeeze(imgoutput[indd,:,:]),kernel)    

if os.path.exists('img_fix_show'+'_'+str(workid)+'_dilate2.mrc'):
    mrc2=mrcfile.open('img_fix_show'+'_'+str(workid)+'_dilate2.mrc',mode='r+')
else:
    mrc2=mrcfile.new('img_fix_show'+'_'+str(workid)+'_dilate2.mrc')
mrc2.set_data(imgoutput_dilate.astype(np.uint8))
mrc2.close()

