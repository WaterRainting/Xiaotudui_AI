#Dataset的这个类，主要用于label和图片的对应标号，以及图片读取等
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir    #目录路径（训练还是验证）
        self.label_dir = label_dir  #标签路径（蚂蚁还是蜜蜂）
        #将两个字符串合并，得到特定label的路径
        self.path=os.path.join(self.root_dir,self.label_dir)
        #列出对应label路径的图片的文件名称，指向一列具体图片
        self.img_path= os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]       #一张图片的文件名称
        #一张图片的路径+名称
        imge_item_path = os.path.join(self.path,img_name)
        img = Image.open(imge_item_path)    #提供一种读取图片的方式
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/train"
ants_label_dir = "ants_image"
bees_label_dir = "bees_image"
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)

img_ants, label_ants= ants_dataset.__getitem__(0)
img_bees, label_bees= bees_dataset.__getitem__(0)
img_bees.show()





