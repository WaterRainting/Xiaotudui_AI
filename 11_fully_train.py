# 完整的模型训练套路
# 导入python库
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.nn import Sequential
from torch import  nn

# 引入模型结构
from fully_train_model import *

# 准备数据集
train_dataset = torchvision.datasets.CIFAR10("./CIFAR10_dataset",train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.CIFAR10("./CIFAR10_dataset",train=False,transform=transforms.ToTensor(),download=True)

# 数据集长度length计算与输出
train_length = len(train_dataset)
test_length = len(test_dataset)
print("训练集数据长度为：{}".format(train_length))
print("测试集数据长度为：{}".format(test_length))

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_dataset,batch_size=64)
test_dataloader = DataLoader(test_dataset,batch_size=64)

# 构建网络模型
tudui = Tudui()

# 损失函数
loss_function = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
learning_rate = 1e-2    #相当于 1×(10)-2=0.01
optimizer = torch.optim.SGD(params=tudui.parameters(),lr=learning_rate)

# tensorboard可视化loss
writer = SummaryWriter("logs")

# 设置训练网络的参数
# 记录训练次数
train_step = 0
# 记录测试次数
test_step = 0
# 训练轮数
epoch = 10

for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+1))

    # 训练步骤开始（需要优化器优化，并计算 Loss）
    tudui.train()   # 有batchnorm和dropout层时必有，开启训练模式，模型会跟踪所有层的梯度，以便在优化器（如 torch.optim.SGD 或 torch.optim.Adam）进行梯度下降时更新模型的权重。
    for data in train_dataloader:
        imgs, targets = data
        output = tudui(imgs)
        loss = loss_function(output,targets)

        # 优化器优化模型
        optimizer.zero_grad()   #梯度清零
        loss.backward()        #反向传播计算梯度
        optimizer.step()        #按照lr进行迭代优化学习

        train_step = train_step + 1
        if train_step % 100 == 0:
            # item()可以以数字形式保存loss，而不是以tensor形式保存loss，此处没区别
            print("训练次数：{}，loss：{}".format(train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),train_step)

    # 测试步骤开始（不需要优化器优化，只需要计算 Loss）
    tudui.eval()    # 有batchnorm和dropout层时必有，开启评估模式，在评估模式下，模型不会跟踪梯度，这有助于减少内存消耗并提高计算效率。评估模式下，Dropout 层会被禁用，所有的神经元都会保留其输出，确保评估时的确定性。
    total_test_loss = 0.0
    total_accuary = 0   # 总准确个数，而不是准确率
    with torch.no_grad(): # 在该模块下，所有"计算"得出的tensor的requires_grad都自动设置为False，但是“赋值”的梯度不受影响
        for data in test_dataloader:
            imgs, targets = data
            output = tudui(imgs)
            loss = loss_function(output,targets)
            total_test_loss = total_test_loss + loss.item()

            # 计算准确个数
            # argmax(1)可以按同一个dimension比较得到，该dimension的最大值的索引，因此索引如果与targets相同就是True，不同就是False
            # 然后用.sum()将True和False加起来，True=1，False=0，因此求和就是正确的个数
            accuary = (output.argmax(1) == targets).sum()
            total_accuary = total_accuary +accuary
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的准确率：{}".format(total_accuary/test_length))
    writer.add_scalar("test_loss",total_test_loss,test_step)
    writer.add_scalar("test_accuary", total_accuary/test_length, test_step)
    test_step = test_step + 1

    # 保存模型参数
    # torch.save(tudui.state_dict(),"tudui_{}.pth".format(i))
    # print("模型已保存")

writer.close()



