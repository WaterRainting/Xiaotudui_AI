#模型保存和加载：已有模型有两种模式，自建模型有一种模式，但是自建可以通过from xxx import *来防止模型复制
import torch
import torchvision
from torch import nn
from model_saveload1 import *

#加载方法1：加载模型
model = torch.load('vgg16_method1.pth')
# print(model)  #这里会打印模型结构

#加载方法2：加载参数
#vgg16 = torch.load("vgg16_method2.pth")    #这里导入的是字典形式
# print(vgg16)                              #这里会以字典形式打印模型参数
vgg16 = torchvision.models.vgg16(False)     #方法2需要先引入结构，然后导入参数
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)                                #这里会打印模型结构

#对于自建模型加载方法1的陷阱
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.module1 = nn.Linear(10,40)
#
#     def forward(self,x):
#         x = self.module1(x)
#         return x
model1 = torch.load("tudui_method1.pth")
print(model1)                               #这里不复制模型过来就会报错，因为需要导入自建模型
#在实际项目中，用from xxx import *来代替直接将模型复制过来，而xxx就是原来模型所在的.py文件名
#但是此时.py文件在import时候要遵循规范的命名格式，特殊符号和数字开头是不被允许的

