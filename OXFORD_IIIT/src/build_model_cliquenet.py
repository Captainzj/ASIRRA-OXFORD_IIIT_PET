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
from OXFORD_IIIT.utils.config import LR, LR_DECAY_EPOCH, NUM_EPOCHS, NUM_IMAGES, MOMENTUM
from OXFORD_IIIT.utils.logger import Logger
import OXFORD_IIIT.cliqueNet_pytorch.cliquenet as cliquenet
from OXFORD_IIIT.utils.custom_plot import plot_acc_loss



def record(filepath, mode, results):
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

    record_file = 'Results/txt/model_breeds_cliquenet_Epoch_' + str(num_epochs) + '_LR_' + str(LR) + '.txt'

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
            record(record_file, 'a', ('{} Loss: {:.8f} Acc: {:.8f}\n'.format(phase, epoch_loss, epoch_acc)))

            if phase == 'train':
                loss_list.append(epoch_loss)
                acc_list.append(epoch_acc)
            else:
                val_loss_list.append(epoch_loss)
                val_acc_list.append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

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

        if (epoch+1) % 10 == 0:
            save_model_path = 'Results/model/build_cliquenet/model_breeds_Epoch_' + str(num_epochs) + '_LR_' + str(LR) + '_' + str(epoch) + '.pkl'
            torch.save(best_model.state_dict(), save_model_path)

        print()
        record(record_file, 'a', '\n')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:8f}\n'.format(best_acc))
    results = ('\nTraining complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)) + ('Best val Acc: {:8f}\n'.format(best_acc))
    record(record_file, 'a', results)

    time_log = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    save_model_path = 'Results/model/build_cliquenet/model_breeds_Epoch_ ' + str(num_epochs) + '_LR_' + str(
        LR) + '_' + time_log + '.pkl'
    torch.save(best_model.state_dict(), save_model_path)

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
                plt.savefig(os.path.join(record_path, "cliqunet_val_sampling_preds.png"))
                return


print('\nProcessing Model Breeds...\n')

logger = Logger('Results/logs')
record_path = 'Results/txt'
classes_breeds = dsets['train'].classes



# cliqueNet
# model = cliquenet.build_cliquenet(input_channels=64, list_channels=[36, 64, 100, 80], list_layer_num=[5, 6, 6, 6], if_att= True)  # block_num = 4 # S0
# model = cliquenet.build_cliquenet(input_channels=64, list_channels=[40, 80, 160, 160], list_layer_num=[6, 6, 6, 6], if_att= True)  # block_num = 4 # S3
model = cliquenet.build_cliquenet(input_channels=64, list_channels=[80, 80, 80], list_layer_num=[6, 6, 6], if_att=True)  # block_num = 3

model = torch.nn.DataParallel(model).cuda()

# load the pre-saved model
#try:
#    last_saved_model = sorted(os.listdir('Results/breeds/model/build_cliquenet'))[-1]
#    load_model_path = 'Results/breeds/model/build_cliquenet/' + last_saved_model
#    if 'pkl' in last_saved_model:
#        model.load_state_dict(torch.load(load_model_path))
#        print('load the saved %s successfully ~' % load_model_path)
#except Exception as e:
#    print(e)
#    pass

if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

visualize_model(model)

# plt.ioff()
# plt.show()
