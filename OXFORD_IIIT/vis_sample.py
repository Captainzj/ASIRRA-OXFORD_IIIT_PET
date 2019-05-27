from PIL import Image
import numpy as np
import torch
import os

def visualize_sample(env_name):

    from visdom import Visdom
    viz = Visdom(env=env_name)
    images_so_far = 0

    val_dir_path = '/home/captain/Desktop/Graduation_Project/OXFORD_IIIT/database/data_breeds/val'
    for val_breeds_dir_name in os.listdir(val_dir_path):  # labels - id

        val_breeds_dir_path = os.path.join(val_dir_path, val_breeds_dir_name)
        for img_filename in os.listdir(val_breeds_dir_path):

            img_path = os.path.join(val_breeds_dir_path,img_filename)
            img_name = ''
            for i, part_name in enumerate(img_filename.split('_')[:-1]):
                if i == 0:
                    img_name += part_name
                else:
                    img_name = img_name+'_'+part_name

            if images_so_far == 1000:
                break# visualize_sample('DenseNet-161-prediction')


            viz.image(torch.from_numpy(np.asarray(
                Image.open(img_path).resize((200, 200), Image.ANTIALIAS))).permute(2, 0, 1),
                      opts=dict(title=img_name))

            images_so_far += 1

visualize_sample('OXFORD-IIIT-PET-samples')
