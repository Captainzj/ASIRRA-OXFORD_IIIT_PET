import copy
import time
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchnet import meter
from torch.autograd import Variable
from tqdm import tqdm
import os

from OXFORD_IIIT.src.data_loader import dset_classes, dset_loaders, dset_sizes, dsets
from OXFORD_IIIT.utils.config import LR, LR_DECAY_EPOCH, NUM_EPOCHS, NUM_IMAGES, MOMENTUM, BATCH_SIZE
from OXFORD_IIIT.utils.logger import Logger
from OXFORD_IIIT.utils.custom_plot import plot_acc_loss
from OXFORD_IIIT.src.densenet_DIY import densenet_DIY
from OXFORD_IIIT.vis_sample import visualize_sample


def record(filepath, mode, results):
    # if not os.path.exists(filepath):
    #     open(filepath,'w')
    with open(filepath, mode) as f:
        f.write(results)

def to_np(x):
    return x.data.cpu().numpy()

def exp_lr_scheduler(optimizer, epoch, init_lr=LR, lr_decay_epoch=LR_DECAY_EPOCH):
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('Learning Rate: {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()

    best_model = model
    best_acc = 0.0
    loss_list, acc_list, val_loss_list, val_acc_list = [], [], [], []

    record_file = 'Results/txt/model_breeds_Epoch_%s_LR_%s_batchSize_%s.txt'% (str(num_epochs),str(LR),str(BATCH_SIZE))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 50)
        record(record_file, 'a',  ('Epoch {}/{}\n'.format(epoch + 1, num_epochs)) + ('--' * 50) + '\n')

        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dset_loaders[phase]):
                '''
                # inputs torch.Size([batch_size, 3, 224, 224])  
                # labels torch.Size([batch_size])
                '''
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.data.item() * inputs.size(0)  # inputs.size(0) 即 batch_size
                running_corrects += (preds == labels).sum().item()

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.8f} Acc: {:.8f}'.format(phase, epoch_loss, epoch_acc))
            record(record_file, 'a', ('{} Loss: {:.8f} Acc: {:.8f}\n'.format(phase, epoch_loss, epoch_acc)) + '\n')

            if phase == 'train':
                loss_list.append(epoch_loss)
                acc_list.append(epoch_acc)
            else:
                val_loss_list.append(epoch_loss)
                val_acc_list.append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

                # time_log = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
                save_model_path = 'Results/model/model_breeds_Epoch_%s_LR_%s_batch_size_%s_.pkl ' % (
                str(num_epochs), str(LR), str(BATCH_SIZE))
                torch.save(best_model.state_dict(), save_model_path)
                print("save_model ... ")

            if phase == 'val':
                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {'loss': epoch_loss, 'accuracy': epoch_acc}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch + 1)

                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), epoch + 1)
                    logger.histo_summary(tag + '/grad', to_np(value.grad), epoch + 1)

                # (3) Log the images
                info = {'images': to_np(inputs.view(-1, 3, 224, 224)[:24])}  # view()创建了一个新的对象，与原对象共享相同的数据 #  R G B
                for tag, inputs in info.items():  # inputs
                    logger.image_summary(tag, inputs, epoch + 1)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:8f}\n'.format(best_acc))
    results = ('\nTraining complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)) + ('Best val Acc: {:8f}\n'.format(best_acc))
    record(record_file, 'a', results)

    plot_path = 'Results/plot/'
    plot_acc_loss(loss_list, val_loss_list, acc_list, val_acc_list, plot_path)

    return best_model

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])  # the same as data_loader config
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # plt.pause(1)

def visualize_model(model, num_images=NUM_IMAGES):
    images_so_far = 0
    plt.figure()

    for _, data in enumerate(dset_loaders['val']):
        inputs, labels = data

        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dset_classes[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                plt.savefig(os.path.join(record_path, "val_sampling_preds.png"))
                return

def visualize_prediction(model, env_name, num_images = 1000):

    from visdom import Visdom
    viz = Visdom(env=env_name)
    images_so_far = 0

    for _, data in enumerate(dset_loaders['val']):

        if images_so_far == num_images:
            break

        inputs, labels = data

        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)  # 若内存不够,就用cpu

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):

            viz.image(inputs.cpu().data[j],  # Val_dataset with No transforms.Normalize
                      opts=dict(title='predicted: {}'.format(dset_classes[preds[j]])))

            images_so_far += 1

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 37),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':

    print('\nProcessing Model Breeds...\n')

    logger = Logger('Results/logs')
    record_path = 'Results/txt'
    classes_breeds = dsets['train'].classes

    model = CNNModel()

    # densenet
    model = torchvision.models.densenet161(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 37)

    # resnet
    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.classifier = nn.Linear(num_ftrs, 37)

    # densenet-DIY
    model = densenet_DIY()

    # load the pre-saved model
    try:
        saved_model = 'Results/model/DenseNet/densenet161/'
        last_saved_model = sorted(os.listdir(saved_model))[-1]
        load_model_path = saved_model + last_saved_model
        if 'pkl' in last_saved_model:
            model.load_state_dict(torch.load(load_model_path))
            print('load the saved %s successfully ~' % load_model_path)
    except Exception as e:
        print(e)
        pass

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

    visualize_prediction(model,'visualize-prediction')
    visualize_model(model)

    plt.ioff()
    plt.show()
