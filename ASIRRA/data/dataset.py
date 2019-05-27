# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class DogCat(data.Dataset):

    def __init__(self, root, fold_th = 4,transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据

        fold_th: 交叉验证的fold序数：0，1，2，3，4
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg 
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.2* fold_th * imgs_num)] + imgs[int(0.2* (fold_th+1) * imgs_num):]
        else:
            self.imgs = imgs[int(0.2* fold_th * imgs_num):int(0.2* (fold_th+1) * imgs_num)]

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])


    def __getitem__(self, index):
        """
        一次返回一张图片的数据

        # test: data/test1/8973.jpg → 8973 （label:id）
        # train: data/train/cat.10004.jpg → cat: 0 dog: 1 (label)
        """
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label, img_path


    def __len__(self):
        return len(self.imgs)
