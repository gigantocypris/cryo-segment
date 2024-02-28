#!/usr/bin/env python
# coding: utf-8


import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from PIL import ImageEnhance


from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from glob import glob
from unet import UNetsmall
from utils import scores
from sklearn.preprocessing import scale
import mrcfile

crf=False
cuda=False
test_type='ciro'
modeltype='unet'
iftureavail=False

model_path_root='checkpoint/'
image_path = '/global/cfs/cdirs/m3562/users/vidyagan/output-cryo-segment/mon_t1_trimmed.rec.nad'

model_list=glob(model_path_root+'*.pth')
model_list.sort()
model_path='checkpoint/checkpoint_450.pth'
testsize=512


    
with open('Zhou_2023/UNET/data/files/labels.txt') as f:
    classes = {}
    for label in f:
        label = label.rstrip().split("\t")
        classes[int(label[0])] = label[1].split(",")[0]

cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

if cuda: 
    current_device = torch.cuda.current_device()
    print("Running on", torch.cuda.get_device_name(current_device))
else:
    print("Running on CPU")

# Configuration

# Label list
with open('Zhou_2023/UNET/data/files/labels.txt') as f:
    classes = {}
    for label in f:
        label = label.rstrip().split("\t")
        classes[int(label[0])] = label[1].split(",")[0]

torch.set_grad_enabled(False)

# Model
model = UNetsmall(n_channels=3, n_classes=3)
state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# Image preprocessing
if iftureavail:
    image = cv2.imread(image_path[:-8]+'true.'+image_path[-3:], cv2.IMREAD_COLOR).astype(float)
    image_original_true = image.astype(np.uint8)
else:
    image_original_true = np.zeros((testsize,testsize,3))


mrc1=mrcfile.open(image_path)
data1=mrc1.data+128

len_frame=data1.shape[0]
ans1_label=np.zeros((len_frame,testsize,testsize)).astype(np.uint8)

ans1_vesicle=ans1_label.copy()
ans1_membrane=ans1_label.copy()
ans1_img=ans1_label.copy()
for index in range(len_frame):
    print(index)
    image= np.tile(np.expand_dims(data1[index,:,:],2),(1,1,3)) 
    
    #image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(float)
    scaley = testsize / image.shape[0]
    scalex = testsize / image.shape[1]
    
    image = cv2.resize(image, dsize=None, fx=scalex, fy=scaley)
    
    #image = 128+40*scale( image[:,:,0], axis=0, with_mean=True, with_std=True, copy=True )
    #image[image>=255]=255
    #image[image<0]=0
    #image=np.tile(np.expand_dims(image,2),[1,1,3])
    
    #enh_sha = ImageEnhance.Sharpness(Image.fromarray(image.astype(np.uint8)))
    #sharpness = 10
    #image_sharped = enh_sha.enhance(sharpness)
    #image = np.asarray(image_sharped)
    contrast = 0
    
    #enh_con = ImageEnhance.Contrast(Image.fromarray(image.astype(np.uint8)))
    #contrast = 5
    #image_contrasted = enh_con.enhance(contrast)
    #image = np.asarray(image_contrasted)
    
    #enh_bri = ImageEnhance.Brightness(Image.fromarray(image.astype(np.uint8)))
    #brightness = 1.5
    #image_brightened = enh_bri.enhance(brightness)
    #image = np.asarray(image_brightened)
    
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
    
    #ans1_prob[index,:,:]=np.squeeze(output_original[0,:,:]*255).numpy().astype(np.uint8)
    ans1_label[index,:,:]=np.squeeze(output_original_label//2*255).numpy().astype(np.uint8)
    ans1_vesicle[index,:,:]=(np.squeeze(output_original_label.numpy()==1)*255).astype(np.uint8)
    ans1_membrane[index,:,:]=(np.squeeze(output_original_label.numpy()==2)*255).astype(np.uint8)
    ans1_img[index,:,:]=(np.squeeze(image[0,0,:,:].numpy())).astype(np.uint8)
#if os.path.exists(image_path.replace('.','_')+'unet_label'+'.mrc'):
#    mrc2=mrcfile.open(image_path.replace('.','_')+'unet_label'+'.mrc',mode='r+')
#else:
#    mrc2=mrcfile.new(image_path.replace('.','_')+'unet_label'+'.mrc')
#mrc2.set_data(ans1_label)
#mrc2.close()

if os.path.exists(image_path.replace('.','_')+'unet_vesicle'+'.mrc'):
    mrc2=mrcfile.open(image_path.replace('.','_')+'unet_vesicle'+'.mrc',mode='r+')
else:
    mrc2=mrcfile.new(image_path.replace('.','_')+'unet_vesicle'+'.mrc')
mrc2.set_data(ans1_vesicle)
mrc2.close()
#
if os.path.exists(image_path.replace('.','_')+'unet_membrane'+'.mrc'):
    mrc2=mrcfile.open(image_path.replace('.','_')+'unet_membrane'+'.mrc',mode='r+')
else:
    mrc2=mrcfile.new(image_path.replace('.','_')+'unet_membrane'+'.mrc')
mrc2.set_data(ans1_membrane)
mrc2.close()

if os.path.exists(image_path.replace('.','_')+'unet_image'+'.mrc'):
    mrc2=mrcfile.open(image_path.replace('.','_')+'unet_image'+'.mrc',mode='r+')
else:
    mrc2=mrcfile.new(image_path.replace('.','_')+'unet_image'+'.mrc')
mrc2.set_data(ans1_img)
mrc2.close()

#if os.path.exists(image_path.replace('.','_')+'unet_prob'):
#    mrc2=mrcfile.open(image_path.replace('.','_')+'unet_prob',mode='r+')
#else:
#    mrc2=mrcfile.new(image_path.replace('.','_')+'unet_prob')
#mrc2.set_data(ans1_prob)
#mrc2.close()
    
            
