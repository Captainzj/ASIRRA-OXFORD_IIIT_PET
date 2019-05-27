'''
@FileName: demo.py
@Author: CaptainSE
@Time: 2019-03-16 
@Desc: 针对 DogVSCats

link: <https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>

.
├── train
└── val

'''


# -*- coding: utf-8 -*-

from __future__ import print_function, division
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
from tqdm import tqdm
from visdom import Visdom

from SRIRRA.config import opt
import argparse
from PIL import Image
from ASIRRA.data.dataset import DogCat
from ASIRRA.utils.custom_plot import plot_acc_loss
from ASIRRA.utils.visualize import Visualizer

parser = argparse.ArgumentParser(description='DogsVSCats_densenet')
parser.add_argument('--model', default='densenet161', type=str, metavar='MODEL',
                    help='select the kind of densenet model')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--device_ids', default='0', type=str, metavar='DEVICE_IDs',
                    help='specified the GPUs device IDs')  # 经测试，若仅想调用一个gpu,需注释掉nn.parallel
args = parser.parse_args()

plt.ion()  # interactive mode

######################################################################
# Training the model


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    vis = Visualizer(opt.env + str(fold_th) + "th")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    loss_list, acc_list, val_loss_list, val_acc_list = [], [], [], []

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        vis.log('Epoch {}/{}'.format(epoch, num_epochs - 1))
        vis.log('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over Data1.
            for inputs, labels, img_path in tqdm(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # outputs = nn.parallel.data_parallel(model, inputs, device_ids=[0])
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)  # preds == labels.data → 0/1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            vis.log('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                loss_list.append(epoch_loss)
                acc_list.append(epoch_acc)
                vis.plot('Train epoch_acc', epoch_acc.item())  # ...
                vis.plot('Train epoch_loss', epoch_loss)  # ...
            else:
                val_loss_list.append(epoch_loss)
                val_acc_list.append(epoch_acc)
                vis.plot('Val epoch_acc', epoch_acc.item())  # ...
                vis.plot('Val epoch_loss', epoch_loss)  # ...

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # best_model = copy.deepcopy(model)

        print()
        vis.log("\n")

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    vis.log('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    vis.log('Best val Acc: {:4f}'.format(best_acc))

    plot_acc_loss(fold_th, loss_list, val_loss_list, acc_list, val_acc_list)

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save the best model_parameters
    torch.save(best_model_wts, os.path.join(opt.save_dir, 'best_model_wts_' + str(fold_th) + '.pth'))
    # # save the best model
    # torch.save(model, os.path.join(opt.save_dir, 'best_model_' + str(fold_th) + '.pth'))

    return model, best_acc


######################################################################
# Test

def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label (cat: 0 dog: 1)'])
        writer.writerows(results)


def test(model):

    since = time.time()
    model.eval()

    test_dataset = DogCat(opt.test_data_root, fold_th, test=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the pre-saved model
    if opt.load_model_path:
        model = torch.load(opt.load_model_path)  # 加载模型（包括结构）
        print('load the saved model successfully ~' + opt.load_model_path)

    viz = Visdom(env="Test-prediction")
    results = []
    images_so_far = 0
    with torch.no_grad():
        for i, (inputs, labels, img_path) in enumerate(test_dataloader):  # labels - id

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probability, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):

                if images_so_far == 1000:
                    break

                viz.image(torch.from_numpy(np.asarray(
                    Image.open(img_path[j]).resize((200, 200), Image.ANTIALIAS))).permute(2, 0, 1),
                           opts=dict(title='predicted: {}'.format(class_names[preds[j]])))

                images_so_far += 1

            batch_results = [(label.item(), preds_.item()) for label, preds_ in zip(labels, preds)]
            results += batch_results

    write_csv(results, opt.result_file)

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return results

######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Generic function to display predictions for a few images
#


def visualize_model(model, num_images=9):

    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.figure()

    viz = Visdom(env="visualize-prediction_" + str(fold_th) + "th")

    with torch.no_grad():
        for i, (inputs, labels, img_path) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            # labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):  # len(inputs)
                images_so_far += 1
                ax = plt.subplot(num_images // 3, 3, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                plt.imshow(np.asarray(Image.open(img_path[j])))

                viz.image(torch.from_numpy(np.asarray(Image.open(img_path[j])
                                                      .resize((200, 200), Image.ANTIALIAS))).permute(2, 0, 1),
                          opts=dict(title='predicted: {}'.format(class_names[preds[j]])))

                if images_so_far == num_images:
                    plt.savefig(opt.logs_dir + "val_sampling_preds_" + str(fold_th) + ".png")
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)


if __name__ == '__main__':

    # #####################################################################
    # Cross validation ...

    best_acc_list = []

    for fold_th in range(opt.fold_num):

        # #####################################################################
        # step1: configure model (Finetuning the convnet)
        #
        if args.model == 'densenet169':
            model_ft = torchvision.models.densenet169(pretrained=True)
        elif args.model == 'densenet121':
            model_ft = torchvision.models.densenet121(pretrained=True)
        elif args.model == 'densenet161':
            model_ft = torchvision.models.densenet161(pretrained=True)
        else:
            model_ft = torchvision.models.densenet201(pretrained=True)

        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, 2)  # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#densenet

        # # load the pre-saved model   # 不加载上一轮的训练模型
        # try:
        #     tmp_ds = opt.save_dir + ".DS_Store"
        #     if os.path.exists(tmp_ds):
        #         os.remove(tmp_ds)
        #     if os.listdir(opt.save_dir):
        #         last_model_wts = sorted(os.listdir(opt.save_dir))[-1]
        #         if 'pth' in last_model_wts:
        #             model_path = os.path.join(opt.save_dir, last_model_wts)
        #             model_ft.load_state_dict(torch.load(model_path))  # 加载模型参数
        #             print('load the saved model successfully ~' + model_path)
        #         else:
        #             print("no saved model.")
        # except Exception as e:
        #     print(e)

        # device = torch.device("cuda:"+ args.device_ids if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_ft = model_ft.to(device)

        # ######################################################################
        # step2: data
        #
        train_dataset = DogCat(opt.train_data_root, fold_th, train=True)
        val_dataset = DogCat(opt.train_data_root, fold_th, train=False)
        image_datasets = {'train': train_dataset, 'val': val_dataset}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], opt.batch_size,
                                                      shuffle=True, num_workers=4)
                       for x in ['train', 'val']}

        dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

        class_names = ['cat', 'dog']  # ...

        # ######################################################################
        # step3: criterion and optimizer
        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), opt.lr, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        # #####################################################################
        # Test
        if args.test == 1:
            test(model_ft)
            break

        # #####################################################################
        # Train and evaluate

        print("Now handling the " + str(fold_th) + "th K-Fold cv....")

        model_ft, best_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, opt.num_epochs)
        best_acc_list.append(best_acc)

        # #####################################################################
        # validate and visualize
        visualize_model(model_ft)

    if args.test == 0:  # not test
        kf_best_acc = 0
        for best_acc in best_acc_list:
            kf_best_acc += best_acc.item()

        print('K-Folds Cross Validation val Acc: {:4f}'.format(kf_best_acc/opt.fold_num))

    pass