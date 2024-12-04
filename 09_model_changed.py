#修改已有的模型，以vgg16为例
import torchvision
from torch import nn

#实例化已有的模型
vgg16_false = torchvision.models.vgg16(False)
print(vgg16_false)

#在最后加一个模块，其中包含一个全连接层
vgg16_false.add_module('add_module_linear',nn.Linear(1000,10))
print((vgg16_false))

#在classifier模块最后加一个全连接层
vgg16_false.classifier.add_module('add_linear',nn.Linear(1000,10))
print(vgg16_false)

#替换classifier模块（6）层为新的全连接层结构
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)
