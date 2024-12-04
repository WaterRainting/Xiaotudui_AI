#使用sequential可以更加快速地搭建神经网络
import torch
from torch import nn
from torch.nn import Sequential,Conv2d,MaxPool2d,Flatten,Linear
from torch.utils.tensorboard import SummaryWriter

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 如果不用sequential就要挨个self.函数，forward中也要挨个调用。有了sequential只需要设定一个model即可，forward中调用model
        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self,x):
        x = self.model1(x)
        return x

tudui = Tudui()
input = torch.ones((64,3,32,32))
output = tudui(input)
print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(tudui,input)   #add_graph可以用来观察模型结构图，参数分别是model和input
writer.close()
