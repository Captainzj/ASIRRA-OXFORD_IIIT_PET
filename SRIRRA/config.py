'''
@FileName: config.py
@Author: CaptainSE
@Time: 2019-03-19 
@Desc: 

'''

# coding:utf8
import os
import warnings
import torch as t


class DefaultConfig(object):

    env = 'DogsVSCats_densenet169-'  # visdom 环境
    vis_port = 8097  # visdom 端口
    model = 'torchvision.models.densenet169'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    fold_num = 1
    train_data_root = './DogsVSCats/train/'  # 训练集存放路径
    test_data_root = './DogsVSCats/test/'  # 测试集存放路径
    # train_data_root = './DogsVSCats_partition/train/'  # 训练集存放路径
    # test_data_root = './DogsVSCats_partition/test/'  # 测试集存放路径

    save_dir = './checkpoints' # 保存训练模型的路径
    logs_dir = './logs/'
    result_file = 'result.csv'
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    # load_model_path = './checkpoints/best_model_0.pth' # 加载预训练的模型的路径，为None代表不加载
    load_model_wts_path = None
    # load_model_wts_path = './checkpoints/best_model_wts_0.pth' # 加载预训练的模型的路径，为None代表不加载

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    batch_size = 96  # batch size

    num_epochs = 15
    lr = 0.001  # initial learning rate


    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()
