#!/usr/bin/env python
# coding: utf-8
#


from __future__ import absolute_import, division, print_function

import click
import cv2

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
from addict import Dict
from glob import glob


from unet import UNet, UNetsmall
from utils import scores#,dense_crf
from sklearn.preprocessing import scale


import pandas as pd


crf=False
cuda=True
#modeltype='unet'
test_type='ciro'
#model_path_root='checkpoint/'+modeltype+'/for'+test_type+'/'
model_path_root='checkpoint/'
model_list=glob(model_path_root+'*.pth')
model_list.sort()


image_path_root='data/'+'val'+'/'
import time
now = int(round(time.time()*1000))
now02 = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
now02=now02.replace(':','_')
now02=now02.replace(' ','-')
result_path_root='result/resultof'+test_type+'/'+now02+'/'
if not(os.path.exists('result/resultof'+test_type+'/')):
    os.mkdir('result/resultof'+test_type+'/')
if not(os.path.exists(result_path_root)):
    os.mkdir(result_path_root)
    

image_list=['100_0_data.png','200_0_data.png','300_0_data.png','400_0_data.png','500_0_data.png','600_0_data.png','700_0_data.png','800_0_data.png','900_0_data.png','1000_0_data.png']
if test_type=='ciro':
    image_list=['100_0_data.png','200_0_data.png','300_0_data.png','400_0_data.png','500_0_data.png','600_0_data.png','700_0_data.png','800_0_data.png','900_0_data.png','1000_0_data.png']


     

config='config/surf.yaml'

cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

if cuda:
    current_device = torch.cuda.current_device()
    print("Running on", torch.cuda.get_device_name(current_device))
else:
    print("Running on CPU")

# Configuration
#CONFIG = Dict(yaml.load(open(config)))

# Label list
with open('data/files/labels.txt') as f:
    classes = {}
    for label in f:
        label = label.rstrip().split("\t")
        classes[int(label[0])] = label[1].split(",")[0]

rows = len(model_list)
cols = (len(classes) + 5) 


        
torch.set_grad_enabled(False)

# Model
model = UNetsmall(n_channels=3, n_classes=3)

for image_path_base in image_list:
    print('test on :'+image_path_base)
    plt.figure(figsize=(30, 4*rows))
    cal_table=[]
    image_path=image_path_root+image_path_base
    i=-1
    ii=-1
    for model_path in model_list:
        i=i+1
        ii=ii+1
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        # Image preprocessing
        image = cv2.imread(image_path[:-8]+'true.'+image_path[-3:], cv2.IMREAD_COLOR).astype(float)
        image_original_true = image.astype(np.uint8)

        
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(float)
#        scaley = CONFIG.IMAGE.SIZE.TEST / image.shape[0]
#        scalex = CONFIG.IMAGE.SIZE.TEST / image.shape[1]
#        
#        image = cv2.resize(image, dsize=None, fx=scalex, fy=scaley)
        image_original = image.astype(np.uint8)
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        image = image.to(device)
        
        # Inference
        output = model(image)
        output_original=output
        output_original_label=np.argmax(output,axis=1)
        outputnumpy=output_original.numpy()
        outputcrf=np.squeeze(np.concatenate(outputnumpy, axis = 1))
        
        #print(output)
        if crf:
            outputsee = dense_crf(image_original, outputcrf)
        else:
            outputsee =outputcrf
        #print(output)
        labelmap = np.argmax(outputsee, axis=0)
        #print(labelmap)
        labels = np.unique(labelmap)
        
        # Show results
        

        ax = plt.subplot(rows, cols, i*cols+1)
        ax.set_title("Input image test for "+os.path.basename(model_path))
        ax.imshow(image_original[:, :, ::-1])
        ax.set_xticks([])
        ax.set_yticks([])
        
        
        ax = plt.subplot(rows, cols, i*cols+2)
        ax.set_title("True image")
        ax.imshow(np.squeeze(image_original_true[:, :, ::-1]))
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = plt.subplot(rows, cols, i*cols+3)
        ax.set_title("p map")
        ax.imshow(np.squeeze(output_original[0,0,:,:]))
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = plt.subplot(rows, cols, i*cols+4)
        ax.set_title("label map before crf")
        ax.imshow(np.squeeze(output_original_label))
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = plt.subplot(rows, cols, i*cols+5)
        ax.set_title("label map")
        ax.imshow(labelmap)
        ax.set_xticks([])
        ax.set_yticks([])
        
        truelabel=np.squeeze(image_original_true[:,:,1])
        
        truelabel2=truelabel//100
        
        
        for j, label in enumerate(labels):
            #print("{0:3d}: {1}".format(label, classes[label]))
            mask = labelmap == label
            ax = plt.subplot(rows, cols, i*cols+j + 6)
            ax.set_title(classes[label])
            ax.imshow(image_original[..., ::-1])
            ax.imshow(mask.astype(np.float32), alpha=0.5, cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
        if crf:
            test_score=scores(truelabel2, labelmap, 3)
            temp=test_score['Class IoU']
            for j in temp:
                test_score['Class IoU: '+str(j)]=temp[j]
            test_score.pop('Class IoU')
            test_score['model_name']=os.path.basename(model_path)
            if len(cal_table)==0:
                cal_table = pd.DataFrame(test_score,index=[ii])
            else:
                cal_table=pd.concat([cal_table,pd.DataFrame(test_score,index=[ii])])
            ii=ii+1    
        
        test_score=scores(truelabel2, np.argmax(outputcrf, axis=0), 3)
        temp=test_score['Class IoU']
        for j in temp:
            test_score['Class IoU: '+str(j)]=temp[j]
        test_score.pop('Class IoU')
        test_score['model_name']=os.path.basename(model_path)+'without crf'
        if len(cal_table)==0:
            cal_table = pd.DataFrame(test_score,index=[ii])
        else:
            cal_table=pd.concat([cal_table,pd.DataFrame(test_score,index=[ii])])    
                

    
    plt.tight_layout()
    plt.savefig(result_path_root+'ans_'+os.path.basename(image_path))
    #plt.show()
    
    cal_table.to_csv(result_path_root+'ans_'+os.path.basename(image_path)[:-4]+'.csv')

file_root='result/resultof'+test_type+'/'
files = os.listdir('result/resultof'+test_type+'/')  # 获取路径下的子文件(夹)列表
for file in files:
    if os.path.isdir(file_root+file):  # 如果是文件夹
        if not os.listdir(file_root+file):  # 如果子文件为空
            os.rmdir(file_root+file)  # 删除这个空文件夹


