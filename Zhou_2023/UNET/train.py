#!/usr/bin/env python
# coding: utf-8
#

import matplotlib
matplotlib.use('Agg')
import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


from unet import UNet, UNetsmall
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from utils.datasets import get_dataset

from matplotlib import pyplot as plt
def train_net(net,
              epochs=5,
              batch_size=1,
              lr=1e-3,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):
    

    def weight_init(m):

        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):

            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()    

    dir_data = 'data/ciro'
    dir_checkpoint = 'checkpoint'
    load_model='checkpoint/checkpoint_60.pth'
    load_model=[]
    ITER_SIZE=10
    ITER_SAVE=50
    DATASET='surf'
    dataset = get_dataset(DATASET)(
    root=dir_data,
    split='train',
    base_size=512,
    crop_size=512,
    mean=(0, 0, 0),
    warp=True,
    scale=[1],
    flip=True,
    )
    
    gpu = gpu and torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")
    
    if load_model:
        state_dict = torch.load(load_model, map_location=lambda storage, loc: storage)
        net.load_state_dict(state_dict)
        print('Model loaded from {}'.format(load_model))
    else:
        net.apply(weight_init)
        
    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
    )
    loader_iter = iter(loader)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(dataset) , str(save_cp), str(gpu)))

    N_train = len(dataset) 

#    optimizer = optim.SGD(net.parameters(),
#                          lr=lr,
#                          momentum=0.9,
#                          weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(),
                          lr=1e-3,
                          betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1,80,20]))
    criterion.to(device)
    loss_historty=[]
    best_loss=1000
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        epoch_loss = 0
        for i in range(1, ITER_SIZE + 1):
            optimizer.zero_grad()
            print(i)
            try:
                images, labels = next(loader_iter)
            except:
                loader_iter = iter(loader)
                images, labels = next(loader_iter)
                
            images = images.to(device)
            labels = labels.to(device)

            print('get image')
            masks_pred = net(images)
            #print (masks_pred.requires_grad)
            #masks_probs_flat = masks_probs.view(-1)
            if gpu:
                true_masks_flat = torch.squeeze(labels.type(torch.cuda.LongTensor))
            else:
                true_masks_flat = torch.squeeze(labels.type(torch.LongTensor))

            loss = criterion(masks_pred, true_masks_flat)
            #loss.requires_grad = True
            
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))
        loss_historty.append(epoch_loss / i)
#        if 1:
#            val_dice = eval_net(net, val, gpu)
#            print('Validation Dice Coeff: {}'.format(val_dice))

        if epoch % ITER_SAVE == 0:
            torch.save(
                net.state_dict(),
                os.path.join(dir_checkpoint, "checkpoint_{}.pth".format(epoch)),
            )
        if epoch_loss/i < best_loss:
            torch.save(
                net.state_dict(),
                os.path.join(dir_checkpoint, "checkpoint_best.pth"),
            )
            best_loss= epoch_loss/i   
             
             
    torch.save(
    net.state_dict(), os.path.join(dir_checkpoint, "checkpoint_final.pth")
    )   
    print(loss_historty)
    print(['best_loss is ',best_loss])
    plt.figure()
    plt.plot(np.log10(loss_historty))
    plt.savefig('training_log.png',dpi=500) 
    plt.figure()
    loss_historty=np.array(loss_historty)
    plt.plot(loss_historty)
    plt.savefig('training.png',dpi=500)

   

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=500, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=20,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=1, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNetsmall(n_channels=3, n_classes=3)
    #net = UNet(n_channels=3, n_classes=3)
#    for para in net.parameters():
#        print(para.requires_grad)
    
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu and torch.cuda.is_available():
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
