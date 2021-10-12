from cv2 import cv2
import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout
import matplotlib.pyplot as plt  # 绘图
from collections import Counter  # 统计频数并且返回字典
import torch
from torch import nn
from torch.nn import ReLU, Sequential, Dropout


class prenn(nn.Module):
    def __init__(self):
        super(prenn, self).__init__()

        # def __init__(self):
        #     super(pre_nn, self).__init__()
        self.model1 = Sequential(
            ReLU(),
            Dropout(p=0.01),
            ReLU(),
            Dropout(p=0.01),
            ReLU(),
            Dropout(p=0.01),
            ReLU(),
            Dropout(p=0.01),
            ReLU(),
            Dropout(p=0.01),
            ReLU(),
            Dropout(p=0.01),
        )

    def forward(self, input):
        return self.model1(input)


def predivt_nn(data, label):
    data = torch.from_numpy(data)
    print('data', data)
    pnn = prenn()
    pnn.train()
    output = pnn(data)
    loss_fn = nn.MSELoss()
    print('output', output)
    optimizer = torch.optim.Adam(pnn.parameters(), lr=0.01)
    loss = loss_fn(output, label)

    # 优化器优化模型
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return pnn
tudui = prenn()
input = torch.ones((64, 3, 32, 32))

pnn = prenn()
pnn.train()
output = tudui(input)
loss_fn = nn.MSELoss()
print('output', output)
optimizer = torch.optim.Adam(pnn.parameters(), lr=0.01)
loss = loss_fn(output, label)

# 优化器优化模型
optimizer.zero_grad()
loss.backward()
optimizer.step()
