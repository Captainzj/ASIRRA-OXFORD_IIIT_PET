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
import OXFORD_IIIT.cliqueNet_pytorch.cliquenet as cliquenet

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

def show_cam_on_image(img_path, mask, model_name):
	img = cv2.imread(img_path)
	height, width, _ = img.shape
	# heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)  # 还原至原图大小,并上色
	heatmap = cv2.applyColorMap(cv2.resize(np.uint8(255*mask), (width, height)), cv2.COLORMAP_JET)  # 还原至原图大小,并上色

	# heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	saved_filepath = os.path.join(img_path.split('.')[0]+'_'+ model_name + '_GRAD_CAM.png')
	cv2.imwrite(saved_filepath, np.uint8(255 * cam))

class FeatureExtractor():
	'''
	Class for extracting activations and registering gradients from targetted intermediate layers
	'''

	def __init__(self, model, target_layers):
		self.model = model
		self.target_layers = target_layers
		self.gradients = []

	def save_gradient(self, grad):  # grad ...
		self.gradients.append(grad)

	def __call__(self, x):
		target_activations = []
		self.gradients = []

		x = self.model.fir_trans(x)
		x = self.model.fir_bn(x)
		x = F.relu(x)
		x = self.model.fir_pool(x)

		feature_I_list = []

		# use stage II + stage II mode
		for i in range(self.model.block_num):
			block_feature_I, block_feature_II = self.model.list_block[i](x)  # block_feature_I 已经与输入x_0拼接
			block_feature_I = self.model.list_compress[i](block_feature_I)  # block_feature_I -> compress

			feature_I_list.append(self.model.list_gb[i](block_feature_I))  # after_compress_block_feature_I -> global-pool
			if i < self.model.block_num - 1:
				x = self.model.list_trans[i](block_feature_II)  # update output (after transition layer) for next Clique_block

		final_feature = feature_I_list[0]  # final_feature of block-one
		for block_id in range(1, len(feature_I_list)):
			final_feature = torch.cat((final_feature, feature_I_list[block_id]), 1)  # concatenate the output of block

		block_feature_I.register_hook(self.save_gradient)
		target_activations += [block_feature_I]

		# final_feature.register_hook(self.save_gradient)
		# target_activations += [final_feature]  # 经过GAP，特征尺寸太小

		# block_feature_II.register_hook(self.save_gradient)
		# target_activations += [block_feature_II]   # 后向传播过程中不经过block_feature_II

		return target_activations, final_feature  # x -- 'last_feature'  # outputs -- 'match_features'


class ModelOutputs():
	'''
	Class for making a forward pass, and getting: (return)
	1. The network output.  # output
	2. Activations from intermeddiate targetted layers.  # target_activations
	3. Gradients from intermeddiate targetted layers.  # self.feature_extractor.gradients
	'''

	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)  # __init__

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, final_feature = self.feature_extractor(x)  # x: feature  # output -- last_feature  torch.Size([1, 512, 7, 7])  / torch.Size([1, 2208, 7, 7])

		final_feature = final_feature.view(final_feature.size()[0], final_feature.size()[1])

		output = self.model.fc(final_feature) # torch.Size([1, 1000])

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

		self.model.fir_trans.zero_grad()
		self.model.fir_bn.zero_grad()
		self.model.fir_pool.zero_grad()
		self.model.list_block.zero_grad()
		self.model.list_trans.zero_grad()
		self.model.list_gb.zero_grad()
		self.model.list_compress.zero_grad()
		self.model.fc.zero_grad()
		one_hot.backward(retain_graph=True)  # retain_graph ...

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy() # grads_val <class 'tuple'>: (1, 512, 14, 14) # gradients torch.Size([1, 512, 14, 14])

		target = features[-1]  # target torch.Size([1, 512, 14, 14]) # features[0] torch.Size([1, 512, 14, 14])
		target = target.cpu().data.numpy()[0, :] # <class 'tuple'>: (512, 14, 14)

		weights = np.mean(grads_val, axis = (2, 3))[0, :] # <class 'tuple'>: (512,)

		cam = np.zeros(target.shape[1 : ], dtype = np.float32)  # <class 'tuple'>: (14, 14)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]  # target <class 'tuple'>: (512, 14, 14)

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)  # 归一化

		return cam

def model_checkpoint_targetLayerName(model, checkpoint_dirpath, target_layer_names):
	# load the pre-saved model
	try:
		last_saved_model = sorted(os.listdir(checkpoint_dirpath))[-1]
		load_model_path = checkpoint_dirpath + last_saved_model
		if 'pkl' in last_saved_model:
			torch.load(load_model_path)
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

	# CliqueNet-18
	model = cliquenet.build_cliquenet(input_channels=64, list_channels=[80, 80, 80], list_layer_num=[6, 6, 6], if_att=True)  # block_num = 3
	checkpoint_dirpath = 'Results/model/build_cliquenet/SVHN-new/'
	target_layer_names = ["fc"]
	grad_cam_CliqueNet_18 = model_checkpoint_targetLayerName(model, checkpoint_dirpath, target_layer_names)

	# CliqueNet-s0
	model = cliquenet.build_cliquenet(input_channels=64, list_channels=[36, 64, 100, 80], list_layer_num=[5, 6, 6, 6],if_att=True)  # block_num = 4 # S0
	checkpoint_dirpath = 'Results/model/build_cliquenet/s0_new/'
	target_layer_names = ["fc"]
	grad_cam_CliqueNet_s0 = model_checkpoint_targetLayerName(model, checkpoint_dirpath, target_layer_names)

	# CliqueNet_S3
	model = cliquenet.build_cliquenet(input_channels=64, list_channels=[40, 80, 160, 160], list_layer_num=[6, 6, 6, 6], if_att= True)  # block_num = 4 # S3
	checkpoint_dirpath = 'Results/model/build_cliquenet/s3_new/'
	target_layer_names = ["fc"]
	grad_cam_CliqueNet_s3 = model_checkpoint_targetLayerName(model, checkpoint_dirpath, target_layer_names)

	grad_cams = {'CliqueNet-18':grad_cam_CliqueNet_18, 'CliqueNet-s0':grad_cam_CliqueNet_s0, 'CliqueNet_S3':grad_cam_CliqueNet_s3}

	# Input (image)
	images = []
	val_dir_path = '/home/captain/Desktop/test-5-27/'

	for img_filename in os.listdir(val_dir_path):
		img_path = os.path.join(val_dir_path, img_filename)
		images.append(img_path)

	# from visdom import Visdom
	# viz = Visdom(env='CliqueNets-Grad-CAM')
	# image_so_far = 0
	for image_path in images:

		# print(image_so_far, image_path)
		# if image_so_far == 100:
		# 	break

		# image_so_far += 1
		img = cv2.imread(image_path, 1)  # <class 'tuple'>: (224, 224, 3)
		img = np.float32(cv2.resize(img, (224, 224))) / 255  # <class 'tuple'>: (224, 224, 3)
		input = preprocess_image(img)  # input torch.Size([1, 3, 224, 224])

		# If None, returns the map for the highest scoring category.
		# Otherwise, targets the requested index.
		target_index = None

		# show_origin_image(image_path)
		for key_i in grad_cams:
			grad_cam = grad_cams[key_i]
			mask = grad_cam(input, target_index)  # __call__
			show_cam_on_image(image_path, mask, key_i)
