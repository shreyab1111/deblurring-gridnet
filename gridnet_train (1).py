# -*- coding: utf-8 -*-
"""Gridnet_train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ecq7DcJeEnt4AKazUuba7oXJ14n_NCbl
"""



import torch
import torch.nn as nn

class lateral_block(nn.Module):

	def __init__(self, inChannels, outChannels, res_conn = True):

		super(lateral_block, self).__init__()

		self.block = nn.Sequential(
				nn.PReLU(),
				nn.Conv2d(inChannels,outChannels,3,1,1),
				nn.PReLU(),
				nn.Conv2d(outChannels,outChannels,3,1,1)

			)
		self.res_conn = res_conn

	def forward(self, x):
		x1 = self.block(x)
		if (self.res_conn):
			x1 = x+x1
		return x1

class down_block(nn.Module):

	def __init__(self, inChannels, outChannels):

		super(down_block, self).__init__()

		self.block = nn.Sequential(
				nn.PReLU(),
				nn.Conv2d(inChannels,outChannels,3,2,1),
				nn.PReLU(),
				nn.Conv2d(outChannels,outChannels,3,1,1)
			)

	def forward(self, x):
		x1 = self.block(x)
		return x1

class up_block(nn.Module):

	def __init__(self, inChannels, outChannels):

		super(up_block, self).__init__()

		self.block = nn.Sequential(
				nn.Upsample(scale_factor=2, mode= 'bilinear'),
				nn.PReLU(),
				nn.Conv2d(inChannels,outChannels,3,1,1),
				nn.PReLU(),
				nn.Conv2d(outChannels,outChannels,3,1,1)
			)

	def forward(self, x):
		x1 = self.block(x)
		return x1

class GridNet(nn.Module):

	def __init__(self, inChannels, outChannels, channel_list=[32,64,96]):

		super(GridNet, self).__init__()

		c1,c2,c3 = channel_list

		self.Lin = lateral_block(inChannels,c1 ,False)

		#row0
		self.L00 = lateral_block(c1 ,c1 )
		self.L01 = lateral_block(c1 ,c1 )
		self.L02 = lateral_block(c1 ,c1 )
		self.L03 = lateral_block(c1 ,c1 )
		self.L04 = lateral_block(c1 ,c1 )

		#row1
		self.L10 = lateral_block(c2,c2)
		self.L11 = lateral_block(c2,c2)
		self.L12 = lateral_block(c2,c2)
		self.L13 = lateral_block(c2,c2)
		self.L14 = lateral_block(c2,c2)

		#row2
		self.L20 = lateral_block(c3,c3)
		self.L21 = lateral_block(c3,c3)
		self.L22 = lateral_block(c3,c3)
		self.L23 = lateral_block(c3,c3)
		self.L24 = lateral_block(c3,c3)

		self.Lout = lateral_block(c1 ,outChannels,False)

		self.d00 = down_block(c1 ,c2)
		self.d01 = down_block(c1 ,c2)
		self.d02 = down_block(c1 ,c2)

		self.d10 = down_block(c2,c3)
		self.d11 = down_block(c2,c3)
		self.d12 = down_block(c2,c3)

		self.u00 = up_block(c2,c1 )
		self.u01 = up_block(c2,c1 )
		self.u02 = up_block(c2,c1 )

		self.u10 = up_block(c3,c2)
		self.u11 = up_block(c3,c2)
		self.u12 = up_block(c3,c2)

	def forward(self,x):

		out_Lin = self.Lin(x)
		out_L00 = self.L00(out_Lin)
		out_L01 = self.L01(out_L00)
		out_L02 = self.L02(out_L01)
		

		out_d00 = self.d00(out_Lin)
		out_d01 = self.d01(out_L00)
		out_d02 = self.d02(out_L01)

		out_L10 = self.L10(out_d00)
		out_L11 = self.L11(out_d01 + out_L10)
		out_L12 = self.L12(out_d02 + out_L11) 

		out_d10 = self.d10(out_d00)
		out_d11 = self.d11(out_L10 + out_d01)
		out_d12 = self.d12(out_L11 + out_d02)

		out_L20 = self.L20(out_d10)
		out_L21 = self.L21(out_d11 + out_L20)
		out_L22 = self.L22(out_d12 + out_L21)

		out_u10 = self.u10(out_L22)
		out_L23 = self.L23(out_L22)
		out_u11 = self.u11(out_L23)
		out_L24 = self.L24(out_L23)
		out_u12 = self.u12(out_L24)

		out_L13 = self.L13(out_u10 + out_L12)
		out_L14 = self.L14(out_u11 + out_L13)
		out_u00 = self.u00(out_u10 + out_L12)
		out_u01 = self.u01(out_u11 + out_L13)
		out_u02 = self.u02(out_u12 + out_L14)

		out_L03 = self.L03(out_u00 + out_L02)
		out_L04 = self.L04(out_u01 + out_L03)

		out_final = self.Lout(out_L04 + out_u02)

		return out_final, out_L04+out_u02

from __future__ import absolute_import, division, print_function
import cv2
import numbers
import math
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from pytorch_msssim import msssim 
import torchvision
from torch.autograd import Variable

from tqdm import tqdm

device = torch.device("cuda:0")
from stacked_DMSHN import stacked_DMSHN
from DMSHN import DMSHN

feed_width = 1024
feed_height =  1024
bokehnet= DMSHN().to(device)


class bokehDataset(Dataset):
    
    def __init__(self, csv_file,root_dir, transform=None):
        
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        bok = pil.open(self.root_dir + self.data.iloc[idx, 0][1:]).convert('RGB')
        org = pil.open(self.root_dir + self.data.iloc[idx, 1][1:]).convert('RGB')
            
        bok = bok.resize((feed_width, feed_height), pil.LANCZOS)
        org = org.resize((feed_width, feed_height), pil.LANCZOS)
        if self.transform : 
            bok_dep = self.transform(bok)
            org_dep = self.transform(org)
        return (bok_dep, org_dep)

transform1 = transforms.Compose(
    [
    transforms.ToTensor(),
])


transform2 = transforms.Compose(
    [
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
])


transform3 = transforms.Compose(
    [
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor(),
])



trainset1 = bokehDataset(csv_file = './data/train.csv', root_dir = '.',transform = transform1)
trainset2 = bokehDataset(csv_file = './data/train.csv', root_dir = '.',transform = transform2)
trainset3 = bokehDataset(csv_file = './data/train.csv', root_dir = '.',transform = transform3)

# batch size changed
trainloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([trainset1,trainset2,trainset3]), batch_size=1,
                                          shuffle=True, num_workers=0)

testset = bokehDataset(csv_file = './data/test.csv',  root_dir = '.', transform = transform1)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=0)


learning_rate = 0.0001
optimizer = optim.Adam( list(bokehnet.parameters()), lr=learning_rate, betas=(0.9, 0.999))
                                                            

sm = nn.Softmax(dim=1)

MSE_LossFn = nn.MSELoss()
L1_LossFn = nn.L1Loss()



def train(dataloader):
    running_l1_loss = 0
    running_ms_loss = 0
    running_sal_loss = 0
    running_loss = 0

    # for i,data in enumerate(tqdm(dataloader),0) : 
    for i,data in enumerate(dataloader,0) :
        bok ,  org = data
        bok ,  org = bok.to(device) , org.to(device)
        bok_pred = bokehnet(org)

        loss = (1-msssim(bok_pred, bok))
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i % 100 == 0):
            print ('Batch: ',i,'/',len(dataloader),' Loss:', loss.item())
        

        if (i%2000==0):
            torch.save(bokehnet.state_dict(), './models/main/dmshn-'+str(epoch)+'-'+str(i)+'.pth')
            print(loss.item())
            
    print (running_l1_loss/len(dataloader))    
    print (running_ms_loss/len(dataloader))  
    print (running_loss/len(dataloader))


def val(dataloader):
    running_l1_loss = 0
    running_ms_loss = 0

    with torch.no_grad():
        for i,data in enumerate(tqdm(dataloader),0) : 
            bok , org = data 
            bok , org = bok.to(device) , org.to(device)
            
            bok_pred = bokehnet(org)
            
            l1_loss = L1_LossFn(bok_pred[0], bok)
            ms_loss = 1-ssim_loss(bok_pred, bok)

            running_l1_loss += l1_loss.item()
            running_ms_loss += ms_loss.item()


    print ('Validation l1 Loss: ',running_l1_loss/len(dataloader))   
    print ('Validation ms Loss: ',running_ms_loss/len(dataloader))


os.makedirs('./models/main/', exist_ok=True)


start_ep = 0
for epoch in range(start_ep,40):    
    print (epoch)
   
    train(trainloader)

    with torch.no_grad():
        val(testloader)

#######################################################################

from __future__ import absolute_import, division, print_function
import cv2
import numbers
import math
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from pytorch_msssim import msssim 
import torchvision
from torch.autograd import Variable

from tqdm import tqdm

device = torch.device("cuda:0")
from gridnet import gridnet

feed_width = 1024
feed_height =  1024
gridnet= gridnet().to(device)


class gridnetDataset(Dataset):
    
    def __init__(self, csv_file,root_dir, transform=None):
        
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        grid = pil.open(self.root_dir + self.data.iloc[idx, 0][1:]).convert('RGB')
        org = pil.open(self.root_dir + self.data.iloc[idx, 1][1:]).convert('RGB')
            
        grid = grid.resize((feed_width, feed_height), pil.LANCZOS)
        org = org.resize((feed_width, feed_height), pil.LANCZOS)
        if self.transform : 
            grid_dep = self.transform(grid)
            org_dep = self.transform(org)
        return (grid_dep, org_dep)

transform1 = transforms.Compose(
    [
    transforms.ToTensor(),
])


transform2 = transforms.Compose(
    [
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
])


transform3 = transforms.Compose(
    [
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor(),
])



trainset1 = gridnetDataset(csv_file = './data/train.csv', root_dir = '.',transform = transform1)
trainset2 = gridnetDataset(csv_file = './data/train.csv', root_dir = '.',transform = transform2)
trainset3 = gridnetDataset(csv_file = './data/train.csv', root_dir = '.',transform = transform3)

# batch size changed
trainloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([trainset1,trainset2,trainset3]), batch_size=1,
                                          shuffle=True, num_workers=0)

testset = gridnetDataset(csv_file = './data/test.csv',  root_dir = '.', transform = transform1)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=0)


learning_rate = 0.0001
optimizer = optim.Adam( list(gridnet.parameters()), lr=learning_rate, betas=(0.9, 0.999))
                                                            

sm = nn.Softmax(dim=1)

MSE_LossFn = nn.MSELoss()
L1_LossFn = nn.L1Loss()



def train(dataloader):
    running_l1_loss = 0
    running_ms_loss = 0
    running_sal_loss = 0
    running_loss = 0

    # for i,data in enumerate(tqdm(dataloader),0) : 
    for i,data in enumerate(dataloader,0) :
        grid ,  org = data
        grid ,  org = grid.to(device) , org.to(device)
        grid_pred = gridnet(org)

        loss = (1-msssim(grid_pred, grid))
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i % 100 == 0):
            print ('Batch: ',i,'/',len(dataloader),' Loss:', loss.item())
        

        if (i%2000==0):
            torch.save(gridnet.state_dict(), './models/main/gridnet-'+str(epoch)+'-'+str(i)+'.pth')
            print(loss.item())
            
    print (running_l1_loss/len(dataloader))    
    print (running_ms_loss/len(dataloader))  
    print (running_loss/len(dataloader))


def val(dataloader):
    running_l1_loss = 0
    running_ms_loss = 0

    with torch.no_grad():
        for i,data in enumerate(tqdm(dataloader),0) : 
            grid , org = data 
            grid , org = grid.to(device) , org.to(device)
            
            grid_pred = gridnet(org)
            
            l1_loss = L1_LossFn(grid_pred[0], grid)
            ms_loss = 1-ssim_loss(grid_pred, grid)

            running_l1_loss += l1_loss.item()
            running_ms_loss += ms_loss.item()


    print ('Validation l1 Loss: ',running_l1_loss/len(dataloader))   
    print ('Validation ms Loss: ',running_ms_loss/len(dataloader))


os.makedirs('./models/main/', exist_ok=True)


start_ep = 0
for epoch in range(start_ep,40):    
    print (epoch)
   
    train(trainloader)

    with torch.no_grad():
        val(testloader)

#######################################################################