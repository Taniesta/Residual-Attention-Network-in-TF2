import os
# import glob
import sys
import datetime

# import imageio
# from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, datasets, Sequential, optimizers

print("%d GPUs are available for tensorflow %s in current environment." % 
      (len(tf.config.experimental.list_physical_devices('GPU')), tf.__version__))

from utils.residual_unit import *
from utils.attention_module import *
from utils.Modules import *

# get the dictionary for the project
pwd = os.getcwd()
sys.path.append(pwd)
# set and create the path for log file for tesnorboard
log_dir = os.path.join(pwd, 'outputs', 'logs')
os.makedirs(log_dir, exist_ok = True)
# set and create the path for saving the images
image_dir = os.path.join(pwd, 'outputs', 'images')
os.makedirs(image_dir, exist_ok = True)
# set and create the path for saving the weights of the model
checkpoint_dir = os.path.join(pwd, 'outputs', 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok = True)

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
structure = 'RAN32'
dataset = 'CIFAR10'
config = "%s-%s-%s" % (structure, dataset, stamp)

# load cifer 10 data
(cifar10_train_x, cifar10_train_y), (cifar10_val_x, cifar10_val_y) = datasets.cifar10.load_data()
print("The shape of CIFAR10 data is: " + str(cifar10_train_x.shape))

# load cifer 10 data
(cifar10_train_x, cifar10_train_y), (cifar10_val_x, cifar10_val_y) = datasets.cifar10.load_data()
print("The shape of CIFAR10 data is: " + str(cifar10_train_x.shape))

cifar10_train_db = tf.data.Dataset.from_tensor_slices((cifar10_train_x, cifar10_train_y)).map(preprocess_cifar10).shuffle(60000)
cifar10_val_db = tf.data.Dataset.from_tensor_slices((cifar10_val_x, cifar10_val_y)).map(preprocess_cifar10)
# get one batch and check the dimension of this batch
cifar10_samples = next(iter(cifar10_val_db.batch(8)))
print("shape of one batch for CIFAR10 images is: %s and %s" % 
      (str(cifar10_samples[0].shape), str(cifar10_samples[1].shape)))

down_3 = [{'filters_residual':[512], 'strides_residual':[1], 'stride_pool':2}]
up_3 = [{'filters_residual':[512], 'strides_residual':[1], 'up_size':2}]

stage_3 = AttentionModule(
    filter_side=[256], stride_side=[1],                   # 16*16*256 -> 16*16*256
    filters_trunk=[256, 512], strides_trunk=[1, 2],       # 16*16*256 -> 16*16*256 -> 8*8*512
    filter_mask=512, s=1, down=down_3, up=up_3,         # 16*16*256 -> 8*8*512
    p = 1, t = 2, r=1)

images = tf.random.normal([1, 16, 16, 256])
stage_3(images).shape
            
        

