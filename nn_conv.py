import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MaxPool2d

input = torch.tensor(([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]]),dtype=torch.float32)
kernel = torch.tensor([[1,2,1],
                      [0,1,0],
                      [2,1,0]])
print(input.shape)
# print(kernel.shape)
#
input = torch.reshape(input,(-1,1,5,5))
# kernel = torch.reshape(kernel,(1,1,3,3))
#
print(input.shape)
# print(kernel.shape)
#
# output1 = F.conv2d(input,kernel,stride=1)
# print(output1)
# print(output1.shape)
# output2 = F.conv2d(input,kernel,stride=2)
# print(output2)
# print(output2.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output =self.maxpool1(input)
        return  output

tudui = Tudui()
output = tudui(input)
print(output)