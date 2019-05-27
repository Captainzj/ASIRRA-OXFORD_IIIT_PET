import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import os
import numpy as np
import argparse
import torch.nn.functional as F
from OXFORD_IIIT.src.densenet_DIY import densenet_DIY_40,densenet_DIY_64,densenet_DIY_CliqueNet_s3,densenet_DIY_100
from OXFORD_IIIT.src.build_model import CNNModel
import OXFORD_IIIT.cliqueNet_pytorch.cliquenet as cliquenet
from OXFORD_IIIT.grad_cam.grad_cam_cliquenet import model_checkpoint_targetLayerName as clique_model_checkpoint_targetLayerName
from OXFORD_IIIT.grad_cam.grad_cam_densenet40 import model_checkpoint_targetLayerName as densenet40_model_checkpoint_targetLayerName

def get_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
	parser.add_argument('--image-path', type=str, default='./database/examples/Birman_3.jpg', help='Input image path')

	args = parser.parse_args()

	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
		print("Using GPU for acceleration")
	else:
		print("Using CPU for computation")

	return args

def preprocess_image(img):
	'''

	:param img: <class 'tuple'>: (224, 224, 3)
	:return:
	'''
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1] # preprocessed_img <class 'tuple'>: (224, 224, 3)
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = np.ascontiguousarray(
		np.transpose(preprocessed_img, (2, 0, 1))) # <class 'tuple'>: (3, 224, 224)
	preprocessed_img = torch.from_numpy(preprocessed_img) # torch.Size([3, 224, 224])
	preprocessed_img.unsqueeze_(0)  # torch.Size([1, 3, 224, 224])
	input = Variable(preprocessed_img, requires_grad = True) # torch.Size([1, 3, 224, 224])
	return input

def show_origin_image(img_path):
	from PIL import Image
	img_name = os.path.split(img_path)[-1].split('.')[0]
	viz.image(torch.from_numpy(np.asarray(
		Image.open(img_path).resize((255, 255), Image.ANTIALIAS))).permute(2, 0, 1),
			  opts=dict(title=img_name))

def show_cam_on_image(img_path, mask, model_name, suffix='.png'):
	img = cv2.imread(img_path)
	height, width, _ = img.shape
	# heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)  # 还原至原图大小,并上色
	heatmap = cv2.applyColorMap(cv2.resize(np.uint8(255*mask), (width, height)), cv2.COLORMAP_JET)  # 还原至原图大小,并上色

	# heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	# saved_filepath = os.path.join(img_path.split('.')[0] + suffix)
	# cv2.imwrite(saved_filepath, np.uint8(255 * cam))

	# from PIL import Image
	# img_name = os.path.split(img_path)[-1].split('.')[0]
	# viz.image(torch.from_numpy(np.asarray(Image.open(img_path).resize((255, 255), Image.ANTIALIAS))).permute(2, 0, 1),opts=dict(title=img_name))
	np_num =  np.uint8(255 * cam)
	torch_num = torch.from_numpy(np.uint8(255 * cam))
	viz.image(torch.from_numpy(cv2.resize(np.uint8(255 * cam), (255, 255))).permute(2, 0, 1),opts=dict(title=model_name))
	# viz.image(torch.from_numpy(cv2.resize(np.uint8(255 * cam), (255, 255))).permute(2, 1, 0),opts=dict(title=model_name))

class FeatureExtractor():
	'''
	Class for extracting activations and registering gradients from targetted intermediate layers
	'''

	def __init__(self, model, target_layers):
		self.model = model
		self.target_layers = target_layers
		self.gradients = []

	def save_gradient(self, grad):  # grad
		self.gradients.append(grad)

	def __call__(self, x):
		outputs = []
		self.gradients = []
		for name, module in self.model._modules.items():
			x = module(x)   #　x 每经过一次moudle()　x.shape都会发生变化　例如, torch.Size([1, 3, 224, 224]) → torch.Size([1, 64, 224, 224])
			if name in self.target_layers:  # match the  targetted intermediate layers
				x.register_hook(self.save_gradient)  # registering gradients from targetted intermediate layers
				outputs += [x]
		return outputs, x  # x -- 'last_feature'  # outputs -- 'match_features'

class ModelOutputs():
	'''
	Class for making a forward pass, and getting: (return)
	1. The network output.  # output
	2. Activations from intermeddiate targetted layers.  # target_activations
	3. Gradients from intermeddiate targetted layers.  # self.feature_extractor.gradients
	'''

	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model.features, target_layers)  # __init__

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output = self.feature_extractor(x)  # x: feature  # output -- last_feature  torch.Size([1, 512, 7, 7])  / torch.Size([1, 2208, 7, 7])
		if 'DenseNet' in str(type(self.model)):
			output = F.relu(output, inplace=True)
			output = F.adaptive_avg_pool2d(output, (1, 1)).view(output.size(0), -1)
		else:
			output = output.view(output.size(0), -1) # torch.Size([1, 25088])
		output = self.model.classifier(output) # torch.Size([1, 1000])

		return target_activations, output # target_activations {list}  target_activations[0] torch.Size([1, 512, 14, 14])  # output torch.Size([1, 1000])

class GradCam():
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names)  # __init__

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):

		'''

		:param input:
		:param index: # If None, returns the map for the highest scoring category.
					# Otherwise, targets the requested index.
		:return:
		'''

		if self.cuda:
			# features -- Activations from intermeddiate targetted layers.(target_activations)  Ex. target_activations[0] torch.Size([1, 512, 14, 14])
			# output -- The network output. (last feature)  Ex. torch.Size([1, 1000])
			features, output = self.extractor(input.cuda())  # __call__
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)  # <class 'tuple'>: (1, 1000)
		one_hot[0][index] = 1  # '激活' 最匹配的unit_index
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.features.zero_grad()  # zero_grad ...
		self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)  #....

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy() # grads_val <class 'tuple'>: (1, 512, 14, 14) # gradients torch.Size([1, 512, 14, 14])

		target = features[-1] # target torch.Size([1, 512, 14, 14]) # features[0] torch.Size([1, 512, 14, 14])  # last_conv
		target = target.cpu().data.numpy()[0, :] # <class 'tuple'>: (512, 14, 14)

		weights = np.mean(grads_val, axis = (2, 3))[0, :] # <class 'tuple'>: (512,) ＃ 基于梯度获取权重!!!
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)  # <class 'tuple'>: (14, 14)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]  # target <class 'tuple'>: (512, 14, 14)   # 加权和的方式得到激活图

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)  # 归一化
		return cam

class GuidedBackpropReLU(Function):

	def forward(self, input):
		positive_mask = (input > 0).type_as(input)
		output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
		self.save_for_backward(input, output)
		return output

	def backward(self, grad_output):
		input, output = self.saved_tensors
		grad_input = None

		positive_mask_1 = (input > 0).type_as(grad_output)
		positive_mask_2 = (grad_output > 0).type_as(grad_output)
		grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

		return grad_input

class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		# # replace ReLU with GuidedBackpropReLU
		# for idx, module in self.model.features._modules.items():
		# 	if module.__class__.__name__ == 'ReLU':
		# 		self.model.features._modules[idx] = GuidedBackpropReLU()

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.forward(input.cuda())
		else:
			output = self.forward(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		# self.model.features.zero_grad()
		# self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)
		# output.backward(gradient=one_hot)

		output = input.grad.cpu().data.numpy()  #<class 'tuple'>: (1, 3, 224, 224)
		output = output[0,:,:,:]

		return output


def model_checkpoint_targetLayerName(model, checkpoint_dirpath, target_layer_names):
	# load the pre-saved model
	try:
		last_saved_model = sorted(os.listdir(checkpoint_dirpath))[-1]
		load_model_path = checkpoint_dirpath + last_saved_model
		if 'pkl' in last_saved_model:
			model.load_state_dict(torch.load(load_model_path))
			print('load the saved %s successfully ~' % load_model_path)
	except Exception as e:
		print(e)
		pass

	model.eval()
	# print(model)

	grad_cam = GradCam(model, target_layer_names, use_cuda=args.use_cuda)

	return grad_cam

if __name__ == '__main__':

	""" 
	python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. 
	"""

	args = get_args()

	# Single_CNN
	model = CNNModel()
	checkpoint_dirpath = 'Results/model/cnn/batchsize_256/'
	target_layer_names = ["31"]  # relu层效果更好
	grad_cam_Single_CNN = model_checkpoint_targetLayerName(model,checkpoint_dirpath,target_layer_names)

	# DenseNet-40
	model = densenet_DIY_40()
	checkpoint_dirpath = 'Results/model/DenseNet/densenet_DIY/depth_40_k_48/'
	target_layer_names = ["norm5"]
	grad_cam_Densenet_40 = densenet40_model_checkpoint_targetLayerName(model,checkpoint_dirpath,target_layer_names)

	# DenseNet-100
	model = densenet_DIY_100()
	checkpoint_dirpath = 'Results/model/DenseNet/densenet_DIY/depth_100_k_32/'
	target_layer_names = ["norm5"]
	grad_cam_Densenet_100 = model_checkpoint_targetLayerName(model, checkpoint_dirpath, target_layer_names)

	# CliqueNet_S3
	model = cliquenet.build_cliquenet(input_channels=64, list_channels=[40, 80, 160, 160], list_layer_num=[6, 6, 6, 6], if_att= True)  # block_num = 4 # S3
	# model = cliquenet.build_cliquenet(input_channels=64, list_channels=[36, 64, 100, 80], list_layer_num=[5, 6, 6, 6], if_att= True)  # block_num = 4 # S0
	checkpoint_dirpath = 'Results/model/build_cliquenet/s3_new/'
	target_layer_names = ["fc"]
	grad_cam_CliqueNet_s3 = clique_model_checkpoint_targetLayerName(model, checkpoint_dirpath, target_layer_names)

	# Densenet161
	model = models.densenet161(pretrained=True)
	num_ftrs = model.classifier.in_features
	model.classifier = torch.nn.Linear(num_ftrs, 37)
	checkpoint_dirpath = 'Results/model/DenseNet/densenet161/'
	target_layer_names = ["norm5"]
	grad_cam_Densenet_161 = model_checkpoint_targetLayerName(model,checkpoint_dirpath,target_layer_names)


	grad_cams = {'Single_CNN': grad_cam_Single_CNN, 'densenet-40': grad_cam_Densenet_40,
				 'DenseNet-100':grad_cam_Densenet_100,'Densenet161':grad_cam_Densenet_161,'CliqueNet-S3':grad_cam_CliqueNet_s3}

	# Input (image)
	images = []
	val_dir_path = '/home/captain/Desktop/Graduation_Project/OXFORD_IIIT/database/data_breeds/val'
	for val_breeds_dir_name in os.listdir(val_dir_path):
		val_breeds_dir_path = os.path.join(val_dir_path, val_breeds_dir_name)
		for img_filename in os.listdir(val_breeds_dir_path):
			img_path = os.path.join(val_breeds_dir_path, img_filename)
			images.append(img_path)

	from visdom import Visdom
	viz = Visdom(env='Models-Grad-CAM')
	image_so_far = 0
	for image_path in images:

		print(image_so_far, image_path)
		if image_so_far == 100:
			break

		image_so_far += 1
		img = cv2.imread(image_path, 1)  # <class 'tuple'>: (224, 224, 3)
		img = np.float32(cv2.resize(img, (224, 224))) / 255  # <class 'tuple'>: (224, 224, 3)
		input = preprocess_image(img)  # input torch.Size([1, 3, 224, 224])

		# If None, returns the map for the highest scoring category.
		# Otherwise, targets the requested index.
		target_index = None

		show_origin_image(image_path)
		for key_i in grad_cams:
			grad_cam = grad_cams[key_i]
			mask = grad_cam(input, target_index)  # __call__
			show_cam_on_image(image_path, mask, key_i)


		# gb_model = GuidedBackpropReLUModel(model, use_cuda=args.use_cuda)
		# gb = gb_model(input, index=target_index)
		# utils.save_image(torch.from_numpy(gb*255), 'gb.jpg')
		#
		# cam_mask = np.zeros(gb.shape)  # <class 'tuple'>: (3, 224, 224)
		# for i in range(0, gb.shape[0]):
		# 	cam_mask[i, :, :] = mask
		#
		# # cam_gb = np.multiply(cam_mask, gb)  # 点乘
		# cam_gb = np.multiply(mask, gb)  # 点乘
		# utils.save_image(torch.from_numpy(cam_gb*255), 'cam_gb.jpg')









