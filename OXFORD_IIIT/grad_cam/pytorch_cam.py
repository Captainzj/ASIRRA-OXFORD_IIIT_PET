# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import requests
from PIL import Image
from torchvision import models, transforms
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import os
import pdb


# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape # <class 'tuple'>: (1, 512, 13, 13)  # <class 'tuple'>: (1, 2208, 7, 7)
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))  # weight_softmax[idx] 1,2208  feature_conv.reshape((nc, h*w) (2208, 49) → cam  49 ...
        cam = cam.reshape(h, w) # (49,) → (7, 7)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min()) # normalize
        cam_img = np.uint8(255 * cam_img)
        # cv2.imwrite("cam_img_No_upsample.jpg", cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))  #
        # cv2.imwrite("cam_img_upsample.jpg", cv2.resize(cam_img, size_upsample))
    return output_cam

## input image
# IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'
# response = requests.get(IMG_URL)
# img_pil = Image.open(io.BytesIO(response.content))
# img_pil.save('test.jpg')

img_fp = 'database/examples/Birman_3.jpg'
img_pil = Image.open(img_fp)

# networks such as googlenet, resnet, densenet already use global average pooling at the end,
# so CAM could be used directly.
model_id = 3
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'  # view net moudel 其中forward()于avgpool前的self.'feature_name'

num_ftrs = net.classifier.in_features
net.classifier = torch.nn.Linear(num_ftrs, 37)

# load the pre-saved model
try:
    last_saved_model = sorted(os.listdir('Results/model/DenseNet/densenet161'))[-1]
    load_model_path = 'Results/model/DenseNet/densenet161/' + last_saved_model
    if 'pkl' in last_saved_model:
        net.load_state_dict(torch.load(load_model_path))
        print('load the saved %s successfully ~' % load_model_path)
except Exception as e:
    print(e)
    pass

net.eval()
# print(net)

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())  # model_id == 1: len: 52  # 3: len:484
weight_softmax = np.squeeze(params[-2].data.numpy())
# weight_softmax <class 'tuple'>: (1000, 512) (imagenet_class_count, squeezenet1_1.hook)
# weight_softmax <class 'tuple'>: (37, 2208) (OXFORD-IIIT PET_class_count, densenet161.hook)

# preprocess
normalize = transforms.Normalize(  # ...  每次运行结果, 分类预测结果不同
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

img_tensor = preprocess(img_pil)  # torch.Size([3, 224, 224])
img_variable = Variable(img_tensor.unsqueeze(0))  # torch.Size([1, 3, 224, 224])
logit = net(img_variable)  # logit torch.Size([1, 1000])  # torch.Size([1, 37])

## download the imagenet category list
#LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
# classes = {int(key):value for (key, value)
#           in requests.get(LABELS_URL).json().items()} # <class 'dict'>:  (1000, ) # {0: 'tench, Tinca tinca', 1: 'goldfish, Carassius auratus', 2: 'great white shark, ...

classes = dict(zip(range(37),['abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'bengal',
                    'birman', 'bombay', 'boxer', 'british_shorthair', 'chihuahua', 'egyptian_mau', 'english_cocker_spaniel',
                    'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond',
                    'leonberger', 'maine_coon', 'miniature_pinscher', 'newfoundland', 'persian', 'pomeranian', 'pug',
                    'ragdoll', 'russian_blue', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'siamese',
                    'sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']))

# 结果有1000(37)类，进行排序
h_x = F.softmax(logit, dim=1).data.squeeze() # h_x torch.Size([1000])
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

# output the prediction
for i in range(0, 5):  # 取前5
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])


# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
# features_blobs[0] <class 'tuple'>: (1, 512, 13, 13) # eatures_blobs[0] <class 'tuple'>: (1, 2208, 7, 7)
# CAMs <class 'list'>  CAMs[0] <class 'tuple'>: (256, 256)

# render the CAM and output
img = cv2.imread(img_fp)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)  # 还原至原图大小,并上色
# cv2.imwrite(img_fp.split('.')[0] +'_heatmap.jpg', heatmap)

# embed origin image
# result = heatmap * 0.3 + img * 0.5
result = heatmap * 0.5 + img * 0.6
cv2.imwrite(img_fp.split('.')[0] +'_CAM.jpg', result)
