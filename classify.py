#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Author: Zhenghao Li
# Date: 2024-11-08

import torch
import torch.nn as nn

class emotionNet(nn.Module):
    def __init__(self, printtoggle):
        super().__init__()

        self.print = printtoggle

        ### feature map layers ###
        # first convolution layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)  # Explicit pooling layer
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128 ,kernel_size=3,stride=1,padding=0 )
        self.pool2= nn.MaxPool2d(2)
        self.act2 = nn.LeakyReLU()
        ### write your codes here ###
        #############################
        # step1:
        # get the input and output channel according to following and above layers
        # after the second block, the ouput size is 11, decide the conv2 and maxpool hyperparameters
        # second layer 

        # third convolution layer
        self.conv3 = nn.Conv2d(128, 256, 4)
        self.pool3 = nn.MaxPool2d(2)
        self.act3 = nn.LeakyReLU()

        # Linear layers
        # step2:
        # calculate the input element numbers and set the first fully connect layer
        self.fc1 = nn.Linear(256 * 4 * 4, 256)

        self.fc1_act = nn.LeakyReLU()
        self.fc2 = nn.Linear(256, 64)
        self.fc2_act = nn.LeakyReLU()
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        #Step 3
        # Set three blocks for forward propagate
        # First block
        # convolution -> maxpool -> relu
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.act1(x)

        # Second block
        # convolution -> maxpool -> relu
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.act2(x)

        # Third block
        # convolution -> maxpool -> relu
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.act3(x)
        # Flatten for linear layers
        x = x.view(x.size(0), -1)

        # fully connect layer
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc2(x)
        x = self.fc2_act(x)
        x = self.fc3(x)

        return x

def makeEmotionNet(printtoggle=False):
    model = emotionNet(printtoggle)
    #loss function
    lossfun = nn.CrossEntropyLoss()
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = .001, weight_decay=1e-5)

    return model, lossfun, optimizer
