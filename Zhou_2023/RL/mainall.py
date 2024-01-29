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
import pickle as pk
from scipy import interpolate 

from toolbox import *


data_path="../../3.Union-find/"
datafilename=data_path+'mon_t2_trimmed_rec_nadunet_membrane_mrc_2D_150_clean.mrc'



iffig=False
iffigv=False

workid=7
if not os.path.exists('results-'+str(workid)):
    os.mkdir('results-'+str(workid))
if not os.path.exists('data-'+str(workid)):
    os.mkdir('data-'+str(workid))
    
    
    

  
np.random.seed ( 30 )
random.seed(139)

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
MAX_T=100000

def reward_cal(robot,action):
 
    dir=COMPASS[ACTION[action]]
    if  robot[0]<0 or robot[0]>=mazedata.shape[0] \
     or robot[1]<0 or robot[1]>=mazedata.shape[1]:
         return(-1)
    if mazedata[robot[0],robot[1]]<1 :
        return(-1)
        
    reward_all=0
    if (dir[0]!=0) and (dir[1]==0) and (robot[0]+dir[0]* foe>=0) and (robot[0]+dir[0]* foe< mazedata.shape[0]):
        x=robot[0]+dir[0]*foe
        y=range(max(robot[1]- foe,0),min(robot[1]+ foe+1, mazedata.shape[1]))
        reward_all+=np.sum(mazevist[x,y]* mazedata[x,y])
    elif (dir[0]==0) and (dir[1]!=0) and (robot[1]+dir[1]* foe>=0) and (robot[1]+dir[1]* foe< mazedata.shape[1]):
        x=range(max(0,robot[0]- foe),min(robot[0]+ foe+1, mazedata.shape[0]))
        y=robot[1]+dir[1]* foe
        reward_all+=np.sum( mazevist[x,y]* mazedata[x,y])
    elif (robot[0]+dir[0]* foe>0) and (robot[0]+dir[0]* foe< mazedata.shape[0]) and (robot[1]+dir[1]* foe>0) and (robot[1]+dir[1]* foe< mazedata.shape[1]):
        reward_all=reward_all- mazedata[robot[0]+dir[0]* foe,robot[1]+dir[1]* foe]* mazevist[robot[0]+dir[0]* foe,robot[1]+dir[1]* foe]
        x=robot[0]+dir[0]* foe
        y=range(max(robot[1]- foe,0),min(robot[1]+ foe+1, mazedata.shape[1]))
        reward_all+=np.sum( mazevist[x,y]* mazedata[x,y])
        x=range(max(0,robot[0]- foe),min(robot[0]+ foe+1, mazedata.shape[0]))
        y=robot[1]+dir[1]* foe
        reward_all+=np.sum( mazevist[x,y]* mazedata[x,y])
    else:
        reward_all=-1
    return(reward_all)

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

MIN_EXPLORE_RATE = 0.3
MIN_LEARNING_RATE = 0.2
DECAY_FACTOR = 10
discount_factor = 0.99
pinf=-10000 


image_path=filename.replace('membrane','image')
mrc1=mrcfile.open(image_path)
data_img=mrc1.data



image_path=datafilename
mrc1=mrcfile.open(image_path)
data1=mrc1.data

px=data1.shape[1:]


indd_list=[100]

#for epl2
indd_list=list(range(120,150))
indd_list.extend(list(range(119,40,-1)))


#for t2
indd_list=list(range(120,135))
indd_list.extend(list(range(119,70,-1)))





indd_list=[86]
ellipse_ans_list=[]
for indd in indd_list:


################################################################################################
#Part 1:  Generate Segments                  
################################################################################################

    
    maze=np.squeeze(data1[indd,:,:])
    maze=maze//255
    plt.figure()
    plt.imshow(np.rot90(np.fliplr(maze),1)) 
    plt.axis('off')
    plt.savefig('results-'+str(workid)+'/'+str(indd)+'_step0_all_'+'.png',dpi=500)


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
    start_pos=tranxy(np.asarray([np.argmax(mazevalue)]),mazedata.shape[1])
    start_pos=(start_pos[0,0],start_pos[0,1])
    mazevist=np.zeros(mazedata.shape)+1
    obv=start_pos
    
    state_0 = obv
    total_reward = 0
    action_vec=np.array([0.0,0.0])
    action_angle=0
    mazevist[state_0[0]-foe:state_0[0]+foe+1,state_0[1]-foe:state_0[1]+foe+1]=0
    current_start_point=obv
    current_js=1
    done=False
    state_all=[]
    state_list=[]
    for t in range(MAX_T):
     
    
        if iffig:
            plt.imshow(np.rot90(np.fliplr(mazevist),1))
            plt.show()        
        state_list.append(state_0)

        best_reward=-1
        best_action=-1
        best_value=-10
        best_angle=0
        reset=False
        for action in range(len(ACTION)):
            state_1=state_0+np.array(COMPASS[ACTION[action]])
            reward=reward_cal(state_1,action) 
            if good_action(action,action_vec):
                if (reward>best_reward):
                    best_action = action
                    best_value=mazevalue[state_1[0],state_1[1]]
                    best_reward=reward
                    best_angle=inner_product(COMPASS[ACTION[action]],action_vec)
                elif  (reward==best_reward) and (reward>=0):
                    if (mazevalue[state_1[0],state_1[1]]>best_value):
                        best_action=action
                        best_value=mazevalue[state_1[0],state_1[1]]
                        best_angle=inner_product(COMPASS[ACTION[action]],action_vec)
                    elif mazevalue[state_1[0],state_1[1]]==best_value:
                        if inner_product(COMPASS[ACTION[action]],action_vec)>best_angle:
                            best_action =action
                            best_angle=inner_product(COMPASS[ACTION[action]],action_vec)
        if best_action == -1:
            reset=True
            for action in range(len(ACTION)):
                state_1=state_0+np.array(COMPASS[ACTION[action]])    
                if (state_1[0]>=0) and (state_1[0]<mazedata.shape[0]) and (state_1[1]>=0) and (state_1[1]<mazedata.shape[1]):
                    if best_action!=-1:
                        best_action=random.sample([best_action,action],1)[0]
                    else:
                        best_action =action
            state=state_0+np.array(COMPASS[ACTION[best_action]])
            action_vec=np.array(COMPASS[ACTION[best_action]])/np.linalg.norm(COMPASS[ACTION[best_action]])
        if best_reward>0.9    :
            action_new=np.array(COMPASS[ACTION[best_action]])/np.linalg.norm(COMPASS[ACTION[best_action]])
            action_vec+=0.1*(action_new-action_vec)
            action_vec=action_vec/np.linalg.norm(action_vec)
            state=state_0+np.array(COMPASS[ACTION[best_action]])
            total_reward += reward
    
        else:
            if current_js==1:
                state=current_start_point
                state_list=state_list[::-1]
                state_list=state_list[:-1]
                current_js=2
                action_vec=np.array([0.0,0.0])
                
            else:
                state_index=state_index+1
                
                state_all.append(state_list)
                start_pos=tranxy(np.asarray([np.argmax(mazevalue*mazevist)]),mazedata.shape[1])
                state=(start_pos[0,0],start_pos[0,1])
                current_start_point=state
                action_vec=np.array([0.0,0.0])
                current_js=1
                state_list=[]
            if (np.sum(mazedata*mazevist)==0) :
                done = True
      
        state_0 = (state[0],state[1])
        robot=state_0
        y=range(max(robot[1]-foe,0),min(robot[1]+foe+1,mazedata.shape[1]))
        x=range(max(0,robot[0]-foe),min(robot[0]+foe+1,mazedata.shape[0]))
        
        for xx in x:
            for yy in y:
                mazevist[xx,yy]=0
                
        if done:
            print("after %f time steps with total reward = %f ."
                  % (t, total_reward))
    
            break
    
    if iffigv:
        for ind,state_list in enumerate(state_all):
            if len(state_list)>10:
                plt.figure()
                plt.imshow(np.rot90(np.fliplr(mazedata),1))
                state_list=np.asarray(state_list)
                plt.plot(state_list[:,0],state_list[:,1])
                plt.savefig('results-'+str(workid)+'/'+str(indd)+'_step1_'+str(ind)+'.png',dpi=500)
                if iffig:
                    plt.show()

    
    save_list=['state_all','mazevalue','mazedata']
    for par in save_list:
        f = open('data-'+str(workid)+'/'+str(indd)+'_'+par,'wb')
        pk.dump(locals()[par],f)
        f.close()
                    
################################################################################################
#Part 2:  RL code for segments                  
################################################################################################
                    
      
        
    ellipse_ans=[]
    i=0
    while i<len(state_all):
        if len(state_all[i])<10:
            del(state_all[i])
        else:
            i=i+1
    
#    for file in glob.glob('results-'+str(workid)+'/'+str(indd)+'_step2_*.png'):
#        os.remove(file)
    
    state_list=np.zeros((MAX_T,3))
    
    state_index=1
    state_i=0
    done=False
    n=len(state_all)
    Q_table_def=np.zeros((2*n,2*n))  
    dir_s=np.zeros((2*n,2))
    dir_e=np.zeros((2*n,2))
    for i in range(n):
        state_now=state_all[i]
        dir_s[2*i]=cal_dir(state_now,'end')   
        dir_s[2*i+1]=cal_dir(state_now[::-1],'end')
        dir_e[2*i]=-dir_s[2*i+1]
        dir_e[2*i+1]=-dir_s[2*i]
    for i in range(2*n):
        for j in range(2*n):
            if i!=j and i//2==j//2:
                Q_table_def[i,j]=pinf
            else:
                s_point=state_all[i//2][-1*((i%2)==0)]
                e_point=state_all[j//2][-1*((j%2)!=0)]
                if cal_dis(s_point,e_point)>150:
                    Q_table_def[i,j]=pinf
    len_list=np.zeros([n,1]) 
    for i in range(n):
        len_list[i]=len(state_all[i])
    notused_bak=np.zeros([n,1])+1 
    notused=notused_bak.copy()  
    while np.sum(notused_bak)>0:
        index_now=np.argmax(notused_bak*len_list)*2
        state_now=state_all[index_now//2]
        notused_bak[index_now//2]=0
           
        current_stater_state=state_now
        current_stater_index=index_now
        reward=np.zeros((2*n,1))
        Q_table=Q_table_def.copy()
        for j in range(2*n):
            if j//2==index_now//2 and j!=index_now:
                reward[j]=1/abs(pinf)
            else:
                s_point=state_all[j//2][-1*(j%2==0)]
                reward[j]=1/abs(cal_jump_cost(dir_s[j,:],s_point,dir_e[index_now,:],state_now[0]))   
                Q_table[j,index_now]=2*(reward[j]**2)
        reward=np.square(reward)*2
        
        if np.argmax(reward)==index_now:
            i=index_now
            notused_bak[i//2]=0
            best_state=state_all[i//2]
            if i%2==1:
                best_state=best_state[::-1]
            state_list[state_i:state_i+len(best_state),0:2]=np.array(best_state)
            state_list[state_i:state_i+len(best_state),2]=state_index
            state_i=state_i+len(best_state)
            state_list[state_i,0:2]=np.array(best_state[0]) 
            state_list[state_i,2]=state_index
            state_i=state_i+1
            state_index=state_index+1
            continue
        
       
        for episode in range(300):
            #print('----------------')
            explore_rate = get_explore_rate(episode)
            learning_rate = get_learning_rate(episode)
            #print(explore_rate)
            index_now=current_stater_index
            index_list=[index_now]
            notused=notused_bak.copy() 
            index_good=index_list.copy()
            val_good=-100
            total_reward=0   
            while True:
                dir_ss=dir_s[index_now]
            
                s_point=state_all[index_now//2][-1*(index_now%2==0)]
                        
                if np.random.random() < explore_rate:
                    action_ca=np.where(Q_table[index_now,:]>pinf)[0]
                    action_ca=action_ca[(notused[action_ca//2]>0).flatten()]
                    if len(action_ca)==0:
                        break
                    action =int(np.random.choice(action_ca,1))

                else:
                    action_ca=np.array(range(2*n))
                    action_ca=action_ca[(notused[action_ca//2]>0).flatten()]
                    if len(action_ca)==0:
                        break
                    action = action_ca[np.argmax(Q_table[index_now,action_ca])]
                    if len(action_ca)==0 or Q_table[index_now,action]==pinf:
                        break

                e_point=state_all[action//2][-1*(action%2!=0)]
                cost=cal_jump_cost(dir_ss,s_point,dir_e[action,:],e_point)
                best_q = np.amax(Q_table[action,:])
                Q_table[index_now,action] += learning_rate * (cost + discount_factor * (best_q) - Q_table[index_now,action])
    
                total_reward+= cost
               

                   
                index_list.append(action)
                index_now=action
                notused[action//2]=0
                # if total_reward+reward[action]>val_good:
                #     val_good= total_reward+reward[action]
                #     index_good=index_list.copy()

        
        explore_rate=0
        index_now=current_stater_index
        index_list=[index_now]
        notused=notused_bak.copy() 
        index_good=index_list.copy()
        val_good=-100
        total_reward=0   
        while True:
            dir_ss=dir_s[index_now]
        
            s_point=state_all[index_now//2][-1*(index_now%2==0)]

            action_ca=np.array(range(2*n))
            action_ca=action_ca[(notused[action_ca//2]>0).flatten()]
            if len(action_ca)==0:
                break
            action = action_ca[np.argmax(Q_table[index_now,action_ca])]
            if len(action_ca)==0 or Q_table[index_now,action]==pinf: #or Q_table[index_now,current_stater_index]>Q_table[index_now,action]:
                break

            
            e_point=state_all[action//2][-1*(action%2!=0)]
            cost=cal_jump_cost(dir_ss,s_point,dir_e[action,:],e_point)
            total_reward+= cost
               
            index_list.append(action)
            index_now=action
            notused[action//2]=0
            if total_reward+reward[action]>val_good and reward[action]>abs(1/pinf):
                val_good= total_reward+reward[action]
                index_good=index_list.copy() 
        for i in index_good:
            notused_bak[i//2]=0
            best_state=state_all[i//2]
            if i%2==1:
                best_state=best_state[::-1]
            state_list[state_i:state_i+len(best_state),0:2]=np.array(best_state)
            state_list[state_i:state_i+len(best_state),2]=state_index
            state_i=state_i+len(best_state)
        if len(index_list)>0:

            state_index=state_index+1
        print(index_list)
    state_list=state_list[:state_i,:]
    
    if iffigv:
        for ind in range(1+int(np.max(state_list[:,2]))):
            ind_list=state_list[:,2]==ind
            if np.sum(ind_list)>20:
                plt.figure()
                plt.imshow(np.rot90(np.fliplr(mazedata),1))
                plt.scatter(state_list[ind_list,0],state_list[ind_list,1],1)
                plt.savefig('results-'+str(workid)+'/'+str(indd)+'_step2_'+str(ind)+'.png',dpi=500)
                if iffig:
                    plt.show()  
                    
################################################################################################
#Part 3:  code for fixing the segments with ellipse。                  
################################################################################################
#    for file in glob.glob('results-'+str(workid)+'/'+str(indd)+'_step3*.png'):
#        os.remove(file)
#    for file in glob.glob('results-'+str(workid)+'/'+str(indd)+'_step4*.png'):
#        os.remove(file)
#    for file in glob.glob('results-'+str(workid)+'/'+str(indd)+'_step5*.png'):
#        os.remove(file)
        
    maze_img=np.squeeze(data_img[indd,:,:])
    plt.figure()
    plt.imshow(np.rot90(np.fliplr(maze_img),1),cmap ='gray')
    plt.axis('off')
    plt.savefig('results-'+str(workid)+'/'+str(indd)+'_step0_img'+'.png',dpi=500)

    
    state_list=state_list.astype(np.int)  
    state_ans=[]
    learningrate=0.2
    
    for ind in range(1,1+int(np.max(state_list[:,2]))):
    
        ind_list=state_list[:,2]==ind
        if np.sum(ind_list)<10:
            continue
        state_list_now=state_list[ind_list,0:2]
        if (np.linalg.norm(state_list_now[0,:]-state_list_now[-1,:])>0) and ( \
           (np.linalg.norm(state_list_now[0,:]-state_list_now[-1,:])/np.sum(ind_list)<0.2) \
           or np.linalg.norm(state_list_now[0,:]-state_list_now[-1,:])<60 ):
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
                    plt.scatter(state_new[:,0],state_new[:,1],1)
                    plt.title('results-'+str(workid)+'/'+str(ind)+'_'+str(dis(state_new[0,:],state_new[-1,:]) ))
                    plt.savefig('results-'+str(workid)+'/'+str(indd)+'_step3_'+str(ind)+'_'+str(ind2)+'.png',dpi=500)
                    #plt.show() 
                

                if  (state_new.shape[0]>20):
                    state_ans.append(state_new)
    robot=[0,0]
    search_range=5
    n=10
    mazedis=np.zeros(mazedata.shape)+10000
    
    
    plt.figure()    
    robot=[0,0]
    search_range=5
    n=10
        
    plt.figure() 
    plt.imshow(np.rot90(np.fliplr(maze_img),1),cmap ='gray')
    imshow=np.zeros((px[1],px[0]))
    ellipse_ans=[]
    for ind,state_new in enumerate(state_ans):
        ellipse = cv2.fitEllipse(np.array(state_new).astype(np.int))
        ellipse_ans.append(ellipse)
        imshow=cv2.ellipse(imshow,ellipse,255,thickness=1)
        plt.scatter(state_new[:,0],state_new[:,1],1,'b',zorder=2)
    plt.axis('off')
    imshow=np.rot90(np.fliplr(imshow),1)    
    see=np.where(imshow>0)
    plt.scatter(see[0],see[1],1,'r',zorder=1)  
    
    plt.savefig('results-'+str(workid)+'/'+str(indd)+'_step4_all_ellipse'+'.png',dpi=500)
    ellipse_ans_list.append([indd,ellipse_ans])
    
f = open('data-'+str(workid)+'/ellipse_ans_list','wb')
pk.dump(ellipse_ans_list,f)
f.close()



    
        