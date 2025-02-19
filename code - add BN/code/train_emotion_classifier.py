#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Author: Zhenghao Li
# Date: 2024-11-08

import torch
import my_net

batchsize    = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader, classes= my_net.utility.loadTrain("./images", batchsize)

model,lossfun,optimizer = my_net.classify.makeEmotionNet(False)

losses, accuracy, _ = my_net.utility.function2trainModel(model, device, train_loader, lossfun, optimizer)

print("--------------------------")
print("Loss and accuracy in every iteration")
for i, (loss, acc) in enumerate(zip(losses, accuracy)):
    print(f"Iteration {i}, loss：{loss:.2f}, accuracy: {acc:.2f}")

PATH = './face_expression.pth'
torch.save(model.state_dict(), PATH)
