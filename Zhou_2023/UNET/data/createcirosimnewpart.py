# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy import misc
from PIL import Image
import cv2
import math
from random import choice
import shutil
import glob

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )
 
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def noise_generator (noise_type,image,para=0):
    """
    Generate noise to a given Image based on required noise type
    
    Input parameters:
        image: ndarray (input image data. It will be converted to float)
        part: parameters
        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    row,col,ch= image.shape
    if noise_type == "gauss":       
        mean = 0.0
        var = 0.01
        if para!=0:
            sigma = para
        else:
            sigma = 0.1    
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        noisy[noisy>255]=255
        noisy[noisy<0]=0
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        s_vs_p = 0.5
        if para!=0:
            amount = para 
        else:
            amount = 0.05
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i , int(num_salt))
              for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i , int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type =="speckle":
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
    else:
        return image

px=512
num_imgs=100
cir_num=20
moti_num=30
r_range=[0.05*px,0.1*px]
indexlist=[100,200,300,400,500,600,700,800,900,1000]
#indexlist=[50]
line_range=[2,8]
color_list=[130,140,160]
inout=[-1,1]
moti_r_range=[2,3]
dir_name='ciro'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
    for file in glob.glob('files/*.txt'):
        shutil.copy(file, dir_name+'/')

    
for index in indexlist:
    for i in range(num_imgs):
        xy=np.floor(np.random.rand(cir_num,2)*px).astype(np.int)
        r=(r_range[0]+r_range[1]*np.random.rand(cir_num,2)).astype(np.int)
        angle=np.random.rand(cir_num,1)*360
        line=np.random.randint(line_range[0],line_range[1]+1,(cir_num,1)).astype(np.int)
        im = np.zeros([px, px], dtype = np.uint8)+148
        im_true= np.zeros([px, px], dtype = np.uint8)
        flagl=np.zeros([cir_num])
        
        for j in range(cir_num):
            r_t=r[j,:]
            flag=True
            for q in range(j):
                if (flagl[q]==1) and (np.linalg.norm([xy[j,0]-xy[q,0],xy[j,1]-xy[q,1]])<1.2*(max(r[j,0],r[j,1])+max(r[q,0],r[q,1]))):
                    flag=False
                    break
            if flag:
                flagl[j]=1
                cv2.ellipse(im,(xy[j,0],xy[j,1]),(r[j,0],r[j,1]),angle[j],0,360,choice(color_list),line[j])
                cv2.ellipse(im_true,(xy[j,0],xy[j,1]),(r[j,0],r[j,1]),angle[j],0,360,255,line[j])

            
                theta=360*np.random.rand(moti_num,1)  
                r_y=r[j,0]*r[j,1]/np.sqrt(np.square(r[j,0]*np.sin((theta-angle[j])*np.pi/180))+np.square(r[j,1]*np.cos((theta-angle[j])*np.pi/180)))
                r_moti=moti_r_range[0]+moti_r_range[1]*np.random.rand(moti_num,1)
                r_y=r_y+np.random.choice(inout,(moti_num,1))*(line[j]+r_moti)*1.2
                x_e=xy[j,0]+np.cos(theta*np.pi/180)*r_y
                y_e=xy[j,1]+np.sin(theta*np.pi/180)*r_y
                
                for q in range(moti_num):
                    cv2.circle(im,(x_e[q],y_e[q]),r_moti[q],choice(color_list),-1)
                    cv2.circle(im_true,(x_e[q],y_e[q]),r_moti[q],100,-1)
                    #cv2.circle(im,(x_e[q],y_e[q]),r_moti[q],255,-1)
            #plt.imshow(im_plot)

        data=np.expand_dims(im,2)
        gauss_im = noise_generator('gauss',data,4)
        sp_im = noise_generator('s&p', gauss_im,0.02)
        #sp_im=gauss_im
        img = Image.fromarray(np.squeeze(sp_im))
        img.save(dir_name+'/'+str(index)+'_'+str(i)+'_data'+'.png')
        img = Image.fromarray(np.squeeze(im_true))
        img.save(dir_name+'/'+str(index)+'_'+str(i)+'_true'+'.png')
        
#        plt.imshow(sp_im[i,:,:],cmap ='gray')
#        plt.savefig('data/'+str(q)+'_'+str(i)+'_data'+'.png')
#        plt.figure(i)
#        plt.imshow(data[i,:,:],cmap ='gray')
#        plt.savefig('data/'+str(q)+'_'+str(i)+'_true'+'.png')
        
        




