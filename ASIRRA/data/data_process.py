'''
@FileName: data_process.py
@Author: CaptainSE
@Time: 2019-03-19
@Desc: 数据预处理

CODE: <https://github.com/ypwhs/dogs_vs_cats/blob/master/Preprocessing%20train%20dataset%20gap.ipynb>
description: <https://ypw.io/dogs-vs-cats-2/>
'''

import os
import shutil


def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


def archive_classname():
    '''
    将不同种类的图片分在不同的文件夹中 by 创建符号链接(symbol link)

    :return:
    '''

    train_filenames = os.listdir('train')  # <class 'list'>
    train_cat = filter(lambda x: x[:3] == 'cat', train_filenames)  # <class 'filter'>
    train_dog = filter(lambda x: x[:3] == 'dog', train_filenames)  # <class 'filter'>

    rmrf_mkdir('train1')
    os.mkdir('train1/cat')
    os.mkdir('train1/dog')

    rmrf_mkdir('test1')
    os.mkdir('test1/cat')
    os.mkdir('test1/dog')

    for filename in train_cat:
        shutil.move('train/' + filename, 'train1/cat/' + filename)

    for filename in train_dog:
        shutil.move('train/' + filename, 'train1/dog/' + filename)

    rmrf_mkdir('test1')
    shutil.move('test/', 'test1/test')


    '''
    .    
    ├── data_process.py
    ├── test [12500 images]
    ├── test1
    │    └── test -> ../test/
    ├── train 
    └── train1  
        ├── cat [12500 images]
        └── dog [12500 images]
    '''

def split_val():
    # 读取文件夹
    train_filenames = os.listdir('train')  # <class 'list'>
    train_cat = filter(lambda x: x[:3] == 'cat', train_filenames)  # <class 'filter'>
    train_dog = filter(lambda x: x[:3] == 'dog', train_filenames)  # <class 'filter'>


    rmrf_mkdir("./DogsVSCats/train")
    os.mkdir('./DogsVSCats/train/cat')
    os.mkdir('./DogsVSCats/train/dog')

    rmrf_mkdir("./DogsVSCats/val")
    os.mkdir('./DogsVSCats/val/cat')
    os.mkdir('./DogsVSCats/val/dog')

    for filename in train_cat:
        if len(filename.split(".")[1]) == 5:
            os.symlink('./train/' + filename, './DogsVSCats/val/cat/' + filename)
        else:
            os.symlink('./train/' + filename, './DogsVSCats/train/cat/' + filename)

    for filename in train_dog:
        if len(filename.split(".")[1]) == 5:
            os.symlink('./train/' + filename, './DogsVSCats/val/dog/' + filename)
        else:
            os.symlink('./train/' + filename, './DogsVSCats/train/dog/' + filename)


if __name__ == '__main__':

    # archive_classname()
    split_val()

    pass
