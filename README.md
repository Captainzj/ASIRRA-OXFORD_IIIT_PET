# ASIRRA-OXFORD_IIIT_PET

## ASIRRA

数据集：[Dogs vs. Cats | Kaggle](<https://www.kaggle.com/c/dogs-vs-cats/data>)

```
DogsVSCats
├── test (12,500 items)
└── train (25,000 items)
```

代码参考：[TRANSFER LEARNING TUTORIAL](<https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>)、[Finetuning Torchvision Models](<https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html>)、 [手把手教你如何在Kaggle猫狗大战冲到Top2%](https://ypw.io/dogs-vs-cats-2/)、[模块化代码结构(内附相关代码)](https://zhuanlan.zhihu.com/p/29024978)、[argparse 使用](http://wiki.jikexueyuan.com/project/explore-python/Standard-Modules/argparse.html)、[Train/Test Split and Cross Validation in Python](https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6)

简要说明：

- ASIRRA/data/dataset.py 👉 训练集：验证集 = 8：2

  

## OXFORD_IIIT_PET

数据集：[get_dataset.sh](<https://github.com/JDonini/Cats-and-Dogs-Classification/blob/master/utils/get_dataset.sh>)

数据预处理：[data_preprocessing.py](<https://github.com/JDonini/Cats-and-Dogs-Classification/blob/master/scripts/breeds/data_preprocessing.py>)

```
database
└── data_breeds
    ├── train
    │   ├── abyssinian
    │   ├── abyssinian.tar.xz
    │   ├── american_bulldog
    │   ├── american_bulldog.tar.xz
    │   ├── american_pit_bull_terrier
    │   ├── basset_hound
    │   ├── beagle
    │   ├── bengal
    │   ├── birman
    │   ├── bombay
    │   ├── boxer
    │   ├── british_shorthair
    │   ├── chihuahua
    │   ├── egyptian_mau
    │   ├── english_cocker_spaniel
    │   ├── english_setter
    │   ├── german_shorthaired
    │   ├── great_pyrenees
    │   ├── havanese
    │   ├── japanese_chin
    │   ├── keeshond
    │   ├── leonberger
    │   ├── maine_coon
    │   ├── miniature_pinscher
    │   ├── newfoundland
    │   ├── persian
    │   ├── pomeranian
    │   ├── pug
    │   ├── ragdoll
    │   ├── russian_blue
    │   ├── saint_bernard
    │   ├── samoyed
    │   ├── scottish_terrier
    │   ├── shiba_inu
    │   ├── siamese
    │   ├── sphynx
    │   ├── staffordshire_bull_terrier
    │   ├── wheaten_terrier
    │   └── yorkshire_terrier
    ├── val
    │   ├── abyssinian
    │   ├── american_bulldog
    │   ├── american_pit_bull_terrier
    │   ├── basset_hound
    │   ├── beagle
    │   ├── bengal
    │   ├── birman
    │   ├── bombay
    │   ├── boxer
    │   ├── british_shorthair
    │   ├── chihuahua
    │   ├── egyptian_mau
    │   ├── english_cocker_spaniel
    │   ├── english_setter
    │   ├── german_shorthaired
    │   ├── great_pyrenees
    │   ├── havanese
    │   ├── japanese_chin
    │   ├── keeshond
    │   ├── leonberger
    │   ├── maine_coon
    │   ├── miniature_pinscher
    │   ├── newfoundland
    │   ├── persian
    │   ├── pomeranian
    │   ├── pug
    │   ├── ragdoll
    │   ├── russian_blue
    │   ├── saint_bernard
    │   ├── samoyed
    │   ├── scottish_terrier
    │   ├── shiba_inu
    │   ├── siamese
    │   ├── sphynx
    │   ├── staffordshire_bull_terrier
    │   ├── wheaten_terrier
    │   └── yorkshire_terrier
├── dataset
│   ├── annotations
│   ├── annotations.tar.gz
│   ├── images
│   └── images.tar.gz 
```



参考：[Cats-and-Dogs-Classification](<https://github.com/JDonini/Cats-and-Dogs-Classification>)、[pytorch-grad-cam](<https://github.com/jacobgil/pytorch-grad-cam/blob/master/grad-cam.py>)



