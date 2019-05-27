'''
@FileName: custom_plot.py
@Author: CaptainSE
@Time: 2019-03-01 
@Desc: 

'''

import matplotlib.pyplot as plt
from visdom import Visdom

from config import opt


# plt.switch_backend('TkAgg')


def plot_acc_loss(fold_th, loss_values, val_loss_values, acc, val_acc):

    vis = Visdom(env="Result_" + str(fold_th))

    # 绘制训练精度和验证精度
    epochs = range(1, len(loss_values) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    vis.matplot(plt)
    # plt.show()
    plt.savefig(opt.logs_dir + "acc_" + str(fold_th) + ".png")

    # 绘制训练损失和验证损失
    # plt.clf() # 清空图像
    plt.figure()
    plt.plot(epochs, loss_values, 'g', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title("Training and Validation loss")
    plt.xlabel('Epochs')
    plt.legend()
    vis.matplot(plt)
    # plt.show()
    plt.savefig(opt.logs_dir + "loss_" + str(fold_th) + ".png")
