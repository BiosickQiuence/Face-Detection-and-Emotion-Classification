#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Author: Zhenghao Li
# Date: 2024-11-08

import torch
import my_net
import matplotlib.pyplot as plt  # 导入绘图库

batchsize = 32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader, classes = my_net.utility.loadTrain("code\\images", batchsize)

model, lossfun, optimizer = my_net.classify.makeEmotionNet(False)

# 记录损失和准确率
losses, accuracy, _ = my_net.utility.function2trainModel(model, device, train_loader, lossfun, optimizer)

print("--------------------------")
print("Loss and accuracy in every iteration")
for i, (loss, acc) in enumerate(zip(losses, accuracy)):
    print(f"Iteration {i}, loss：{loss:.2f}, accuracy: {acc:.2f}")

# 绘制损失曲线和准确率曲线
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.title('Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(accuracy, label='Accuracy', color='orange')
plt.title('Accuracy Curve')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()  # 自适应布局
plt.show()  # 显示图表

# 保存模型
PATH = 'code\\face_expression.pth'
torch.save(model.state_dict(), PATH)
