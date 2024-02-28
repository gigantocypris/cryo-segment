#!/usr/bin/env python
# coding: utf-8
#


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
outputname=image_path[:-len(os.path.basename(image_path))]+'/test1'
model_list=glob(model_path_root+'*.pth')
model_list.sort()
model_path='checkpoint/checkpoint_450.pth'
testsize=512
index=100
if not(os.path.exists(outputname)):
    os.mkdir(outputname)
    
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
#model = UNet(n_channels=3, n_classes=3)
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
    outputsee = outputcrf
#print(output)
labelmap = np.argmax(outputsee, axis=0)
#print(labelmap)
labels = np.unique(labelmap)

# Show results
rows = int(np.floor(np.sqrt(len(labels) )))
cols = int(np.ceil((len(labels) + 5) / rows))

plt.figure(figsize=(30, 10))
ax = plt.subplot(rows, cols, 1)
ax.set_title("Input image")
ax.imshow(image_original[:, :, ::-1])
ax.set_xticks([])
ax.set_yticks([])


ax = plt.subplot(rows, cols, 2)
ax.set_title("True image")
ax.imshow(image_original_true[:, :, ::-1])
ax.set_xticks([])
ax.set_yticks([])

ax = plt.subplot(rows, cols, 3)
ax.set_title("p map")
ax.imshow(np.squeeze(output_original[0,0,:,:]))
ax.set_xticks([])
ax.set_yticks([])

ax = plt.subplot(rows, cols, 4)
ax.set_title("label map before crf")
ax.imshow(np.squeeze(output_original_label))
ax.set_xticks([])
ax.set_yticks([])

ax = plt.subplot(rows, cols, 5)
ax.set_title("label map")
ax.imshow(labelmap)
ax.set_xticks([])
ax.set_yticks([])

truelabel=np.squeeze(image_original_true[:,:,1])

truelabel2=truelabel//100


if iftureavail:
    print(scores(truelabel2, labelmap, 3))

for i, label in enumerate(labels):
    print("{0:3d}: {1}".format(label, classes[label]))
    mask = labelmap == label
    ax = plt.subplot(rows, cols, i + 6)
    ax.set_title(classes[label])
    ax.imshow(image_original[..., ::-1])
    ax.imshow(mask.astype(np.float32), alpha=0.5, cmap="viridis")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('result/vis/contrast='+str(contrast)+'_'+str(index)+'_'+os.path.basename(image_path.replace('.','_')))
plt.show()

img = Image.fromarray(np.squeeze(image_original[:, :, ::-1]))
img.save(outputname+'/'+os.path.basename(image_path).replace('.','_')+'_'+str(index)+'_data'+'.png')
#img = Image.fromarray(np.squeeze(output_original[0,0,:,:]*255).numpy().astype(np.uint8))
#img.save(outputname+'/'+os.path.basename(image_path).replace('.','_')+'_'+str(index)+'_out'+'.png')
img = Image.fromarray(np.squeeze(output_original_label.numpy()*127).astype(np.uint8))
img.save(outputname+'/'+os.path.basename(image_path).replace('.','_')+'_'+str(index)+'_outlabel'+'.png')
img = Image.fromarray(((np.squeeze(output_original_label.numpy())==1)*255).astype(np.uint8))
img.save(outputname+'/'+os.path.basename(image_path).replace('.','_')+'_'+str(index)+'_out_vesicle'+'.png')
img = Image.fromarray(((np.squeeze(output_original_label.numpy())==2)*255).astype(np.uint8))
img.save(outputname+'/'+os.path.basename(image_path).replace('.','_')+'_'+str(index)+'_out_membrane'+'.png')

#np.savetxt(outputname+'/'+os.path.basename(image_path).replace('.','_')+'_'+str(index)+'_out'+'.csv',np.squeeze(output_original) , delimiter = ',')
