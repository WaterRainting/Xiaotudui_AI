#loss损失函数，backward反向传播（计算梯度），optim用于更新参数（优化）
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Sequential,Conv2d,MaxPool2d,Flatten,Linear
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10("./CIFAR10_dataset",train=False,transform=transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=1)

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

loss = nn.CrossEntropyLoss()    #定义loss function
tudui = Tudui()
optim = torch.optim.SGD(params=tudui.parameters(),lr=0.01)  #定义优化器，函数参数(模型的参数，学习率，...)

#最内层一次循环是一组batch，遍历所有batch；最外层一次循环是所有batch（一次epoch），也就是dataset的所有数据
#因此只有外层循环的每次循环，代表数据的一次完整训练，因此外次循环的次数就是所有数据的总训练次数
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, target = data
        optim.zero_grad()                       # 梯度归零，防止上次循环的梯度干扰
        outputs = tudui(imgs)
        data_loss = loss(outputs, target)       # 计算loss大小
        data_loss.backward()                    # 反向传播法计算梯度
        optim.step()                            # 更新参数
        running_loss = running_loss + data_loss #将每一组batch的loss求和得到完整数据的loss
    print(running_loss)                         #训练时这个running_loss理应不断减小

