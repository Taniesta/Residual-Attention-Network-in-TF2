# import os
# import sys
# import datetime

# import imageio
# from PIL import Image
# import matplotlib.pyplot as plt

# import numpy as np
# import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras import layers, datasets, Sequential, optimizers

# print("%d GPUs are available for tensorflow %s in current environment." % 
#       (len(tf.config.experimental.list_physical_devices('GPU')), tf.__version__))

from utils.residual_unit import *
from utils.attention_module import *
from utils.models import Attention56

# # get the dictionary for the project
# pwd = os.getcwd()
# sys.path.append(pwd)
# # set and create the path for log file for tesnorboard
# log_dir = os.path.join(pwd, 'outputs', 'logs')
# os.makedirs(log_dir, exist_ok = True)
# # set and create the path for saving the images
# image_dir = os.path.join(pwd, 'outputs', 'images')
# os.makedirs(image_dir, exist_ok = True)
# # set and create the path for saving the weights of the model
# checkpoint_dir = os.path.join(pwd, 'outputs', 'checkpoints')
# os.makedirs(checkpoint_dir, exist_ok = True)

# stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# structure = 'RAN32'
# dataset = 'CIFAR10'
# config = "%s-%s-%s" % (structure, dataset, stamp)

# # normalize data
# def preprocess_cifar10(x, y):        
#     x = 2*tf.cast(x, dtype=tf.float32)/255 - 1
#     y = tf.squeeze(tf.one_hot(y, depth=10, dtype=tf.int32))
#     return x, y

# # load cifer 10 data
# (cifar10_train_x, cifar10_train_y), (cifar10_val_x, cifar10_val_y) = datasets.cifar10.load_data()
# print("The shape of CIFAR10 data is: " + str(cifar10_train_x.shape))

# # load cifer 10 data
# (cifar10_train_x, cifar10_train_y), (cifar10_val_x, cifar10_val_y) = datasets.cifar10.load_data()
# print("The shape of CIFAR10 data is: " + str(cifar10_train_x.shape))

# cifar10_train_db = tf.data.Dataset.from_tensor_slices((cifar10_train_x, cifar10_train_y)).map(preprocess_cifar10).shuffle(60000)
# cifar10_val_db = tf.data.Dataset.from_tensor_slices((cifar10_val_x, cifar10_val_y)).map(preprocess_cifar10)
# # get one batch and check the dimension of this batch
# cifar10_samples = next(iter(cifar10_val_db.batch(8)))
# print("shape of one batch for CIFAR10 images is: %s and %s" % 
#       (str(cifar10_samples[0].shape), str(cifar10_samples[1].shape)))

model = Attention56()
model.build(input_shape=(None, 32, 32, 3))
print(model.summary())

# res = ResidualUnit(128)
# model = Sequential()
# model.add(res)
# model.build(input_shape=(None, 32, 32, 3))
# print(model.summary())