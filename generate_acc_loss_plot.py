def extract_from_txt(txt_filepath):

	file_obj = open(txt_filepath)

	all_lines = file_obj.readlines()

	train_Loss_list, train_Acc_list, val_Loss_list, val_Acc_list = [], [], [], []

	for line in all_lines:
		line = line[:-1]  # 忽略换行符
		if 'train' in line:
			train_Loss_list.append(line[12:22])	
			train_Acc_list.append(line[28:])
			
		if 'val' in line and 'Best' not in line:
			val_Loss_list.append(line[10:20])	
			val_Acc_list.append(line[26:])

	file_obj.close()
	
	return train_Loss_list,train_Acc_list, val_Loss_list, val_Acc_list

def str_list_2_float_list(str_lists):
	float_lists = []
	for str_list in str_lists:
		float_list = [float(i) for i in str_list]
		float_lists.append(float_list)
	return float_lists

def generate_visdom_line(log_filepath,env_name,epoch, title=''):

	log_filepath = log_filepath
	train_Loss_list ,train_Acc_list, val_Loss_list, val_Acc_list = extract_from_txt(log_filepath)

	# print(train_Loss_list, '\n', train_Acc_list, '\n', val_Loss_list, '\n', val_Acc_list)

	from visdom import Visdom
	import numpy as np
	viz = Visdom(env=env_name)	#设置环境窗口的名称是'my_wind',如果不设置名称就在main中

	float_lists = str_list_2_float_list([train_Loss_list[:32], train_Acc_list[:32], val_Loss_list[:32], val_Acc_list[:32]])  # Default: Epoch-32
	viz.line(Y=np.array(float_lists[0]),X=list(range(1,epoch+1)), opts=dict(title='Train_Loss',xlabel = 'Epoch', ylabel = 'loss'))
	viz.line(Y=np.array(float_lists[1]),X=list(range(1,epoch+1)), opts=dict(title='Train_Accuracy',xlabel = 'Epoch', ylabel = 'acc'))
	viz.line(Y=np.array(float_lists[2]),X=list(range(1,epoch+1)), opts=dict(title='Val_Loss',xlabel = 'Epoch', ylabel = 'loss'))
	viz.line(Y=np.array(float_lists[3]),X=list(range(1,epoch+1)), opts=dict(title='Val_Accuracy',xlabel = 'Epoch', ylabel = 'acc'))

	viz.line(Y=np.column_stack((np.array(float_lists[0]),np.array(float_lists[2]))),X=list(range(1,epoch+1)), opts=dict(title='Loss',xlabel = 'Epoch', ylabel = 'loss',legend=['train_loss', 'val_loss']))
	viz.line(Y=np.column_stack((np.array(float_lists[1]),np.array(float_lists[3]))),X=list(range(1,epoch+1)), opts=dict(title='Accuracy',xlabel = 'Epoch', ylabel = 'acc',legend=['train_acc', 'val_acc']))

def generate_Simple_CNN_plot():
	### Simple-CNN ####
	# dir_path = '/home/captain/Documents/code/Cats-and-Dogs-Classification-master/models/breeds/txt/'
	dir_path = './OXFORD_IIIT/Results/logs/'

	log_filepath_1 = dir_path + "model_breeds_CNN_Epoch_32_LR_0.01_batchSize_24.txt"
	env_name_1  = 'Simple-CNN_batchSize-24'
	epoch_1 = 32
	generate_visdom_line(log_filepath_1, env_name_1, epoch_1)

	log_filepath_2 = dir_path + "model_breeds_CNN_Epoch_32_LR_0.01_batchSize_128.txt"
	env_name_2 = 'Simple-CNN_batchSize-128'
	epoch_2 = 32
	generate_visdom_line(log_filepath_2, env_name_2, epoch_2)

	log_filepath_3 = dir_path + "model_breeds_CNN_Epoch_32_LR_0.01_batchSize_256.txt"
	env_name_3 = 'Simple-CNN_batchSize-256'
	epoch_3 = 32
	generate_visdom_line(log_filepath_3, env_name_3, epoch_3)

def generate_DenseNet_ImageNet_plot():
	# ### Densenet-ImageNet ####
	# dir_path = '/home/captain/Documents/code/Cats-and-Dogs-Classification-master/models/breeds/txt/'
	dir_path = './OXFORD_IIIT/Results/logs/'

	log_filepath = dir_path + "model_breeds_densenet121_Epoch_32_LR_0.01_batchSize_24.txt"
	env_name  = 'DenseNet_densenet-121'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

	log_filepath = dir_path+ "model_breeds_densenet161_Epoch_32_LR_0.01_batchSize_24.txt"
	env_name  = 'DenseNet_densenet-161'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

	log_filepath = dir_path+ "model_breeds_densenet169_Epoch_32_LR_0.01_batchSize_24.txt"
	env_name  = 'DenseNet_densenet-169'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

	log_filepath = dir_path+ "model_breeds_densenet201_Epoch_32_LR_0.01_batchSize_24.txt"
	env_name  = 'DenseNet_densenet-201'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

def generate_DenseNet40_plot():

	# ### Densenet-40 ####
	# dir_path = '/home/captain/Documents/code/Cats-and-Dogs-Classification-master/models/breeds/txt/'
	dir_path = './OXFORD_IIIT/Results/logs/'

	log_filepath = dir_path + "model_breeds_densenet_DIY_depth_40_k_12_Epoch_32_LR_0.01_batchSize_128.txt"
	env_name = 'DenseNet40_Growth Rate-12'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

	log_filepath = dir_path+ "model_breeds_densenet_DIY_depth_40_k_24_Epoch_32_LR_0.01_batchSize_96.txt"
	env_name = 'DenseNet40_Growth Rate-24'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

	log_filepath = dir_path+ "model_breeds_densenet_DIY_depth_40_k_32_Epoch_32_LR_0.01_batchSize_64.txt"
	env_name = 'DenseNet40_Growth Rate-40'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

	log_filepath = dir_path+ "model_breeds_densenet_DIY_depth_40_k_48_Epoch_32_LR_0.01_batchSize_56.txt"
	env_name = 'DenseNet40_Growth Rate-48'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

	log_filepath = dir_path+ "model_breeds_densenet_DIY_depth_40_k_64_Epoch_32_LR_0.01_batchSize_32.txt"
	env_name = 'DenseNet40_Growth Rate-64'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

def generate_DenseNetDIY_plot():

	# ### Densenet-40 ####
	# dir_path = '/home/captain/Documents/code/Cats-and-Dogs-Classification-master/models/breeds/txt/'
	dir_path = './OXFORD_IIIT/Results/logs/'

	log_filepath = dir_path+ "model_breeds_densenet_DIY_depth_40_k_48_Epoch_32_LR_0.01_batchSize_56.txt"
	env_name  = 'DenseNet-DIY_Depth-40'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

	log_filepath = dir_path+ "model_breeds_densenet_DIY_depth_64_k_48_Epoch_32_LR_0.01_batchSize_40.txt"
	env_name = 'DenseNet-DIY_Depth-64'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

	log_filepath = dir_path+ "model_breeds_densenet_DIY_depth_100_k_48_Epoch_32_LR_0.01_batchSize_14.txt"
	env_name = 'DenseNet-DIY_Depth-100'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

def generate_CliqueNet():
	# dir_path = '/home/captain/Documents/code/Cats-and-Dogs-Classification-master/models/breeds/txt/'
	dir_path = './OXFORD_IIIT/Results/logs/'

	# log_filepath = dir_path + "model_breeds_cliquenet_blocknum_3_Epoch_100_LR_0.01.txt"
	# env_name = 'CliqueNet_T-18-k-80'
	# epoch = 32
	# generate_visdom_line(log_filepath, env_name, epoch)
	#
	# log_filepath = dir_path + "model_breeds_cliquenet_blockNum_4_Epoch_32_LR_0.01.txt"
	# env_name = 'CliqueNet_T-24-k-[40,80,160,160]'
	# epoch = 32
	# generate_visdom_line(log_filepath, env_name, epoch)
	#
	# log_filepath = dir_path + "model_breeds_cliquenet_blocknum_4_S0_Epoch_32_LR_0.01.txt"
	# env_name = 'CliqueNet_T-23-k-[36,64,100,80]'
	# epoch = 32
	# generate_visdom_line(log_filepath, env_name, epoch)

	log_filepath = dir_path + "model_breeds_cliquenet_SVHN_new_Epoch_32_LR_0.01_batchSize_16.txt"
	env_name = 'CliqueNet_SVHN'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

	log_filepath = dir_path + "model_breeds_cliquenet_s0_new_Epoch_32_LR_0.01_batchSize_24.txt"
	env_name = 'CliqueNet_ImageNet_s0'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

	log_filepath = dir_path + "model_breeds_cliquenet_s3_new_Epoch_32_LR_0.01_batchSize_22.txt"
	env_name = 'CliqueNet_ImageNet_s3'
	epoch = 32
	generate_visdom_line(log_filepath, env_name, epoch)

# generate_Simple_CNN_plot()
# generate_DenseNet_ImageNet_plot()
# generate_DenseNet40_plot()
# generate_DenseNetDIY_plot()
generate_CliqueNet()


