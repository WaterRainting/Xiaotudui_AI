#模型保存和加载：已有模型有两种模式，自建模型有一种模式，但是自建可以通过from xxx import *来防止模型复制
import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(False)

#保存方法1：模型结构+模型参数
torch.save(vgg16,"vgg16_method1.pth")

#保存方法2：模型参数（以字典形式保存，可以减小模型结构带来的存储空间）（官方推荐方法）
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

#方法1的一个自建模型的陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.module1 = nn.Linear(10,40)

    def forward(self,x):
        x = self.module1(x)
        return x

tudui = Tudui()
torch.save(tudui,"tudui_method1.pth")


