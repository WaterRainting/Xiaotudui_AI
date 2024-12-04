# 是11_fully_train的模型保存，一般在项目中都将模型单独保存到一个.py文件中
import torch
from torch.nn import Sequential
from torch import  nn

# 搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.module = Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.module(x)
        return x

# 用于验证在正确输入格式的情况下是否得到正确输出格式，即验证模型结构
if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64,3,32,32))
    output = tudui(input)
    print(output.shape)
