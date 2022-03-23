# -*- coding: utf-8 -*-

import os
import sys
import random
import warnings
import cv2
import pickle
import numpy as np
#import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Add, UpSampling2D
from keras.layers import Input,Concatenate, UpSampling2D,Activation, BatchNormalization, LeakyReLU, Add, ZeroPadding2D, ReLU

from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.optimizers import Adam
#import tensorflow as tf
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow import keras
import keras
from PIL import Image
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array


from attn_augconv import augmented_conv2d
#from attn_augconv2D import augmented_conv2d
import tensorflow as tf

#%%
IMG_HEIGHT = " Hight of the Image"
IMG_WIDTH = " Width of the Image"
IMG_CHANNELS = "Depth of the Image"



def bn_act(x, act=True):
    x = keras.layers.BatchNormalization()(x)
    if act == True:
        x = keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = keras.layers.SeparableConv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.SeparableConv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = keras.layers.SeparableConv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    res = augmented_conv2d(res, filters, kernel_size=3, strides = 1,
                        depth_k=4, depth_v=4, num_heads=2, relative_encodings=True)
    
    #res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = keras.layers.SeparableConv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = keras.layers.UpSampling2D((2, 2))(x)
    c = keras.layers.Concatenate()([u, xskip])
    return c

def AA_ResUNet():
    f = [32, 64, 128, 256]
    input_shape=(IMG_HEIGHT, IMG_WIDTH,  IMG_CHANNELS)
    inputs = keras.layers.Input(input_shape)
    
    #inputs = keras.layers.Input((256, 256, 3))
    ## Encoder
    e0 = inputs
    #e1 = stem(e0, f[0]) #64
    e1 = residual_block(e0, f[0], strides=1)
    print(e1.shape)
    e2 = residual_block(e1, f[1], strides=2) #128
    print(e2.shape)
    e3 = residual_block(e2, f[2], strides=2) #256
    print(e3.shape)
    #e4 = residual_block(e3, f[3], strides=2)
    #e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    # b0 = conv_block(e3, f[3], strides=2) #512
    # b1 = conv_block(b0, f[3], strides=1)
    b1 = residual_block(e3, f[3], strides=2)
    print(b1.shape)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e3)
    d1 = residual_block(u1, f[2])
    print(d1.shape)
    
    u2 = upsample_concat_block(d1, e2)
    d2 = residual_block(u2, f[1])
    print(d2.shape)
    
    u3 = upsample_concat_block(d2, e1)
    d3 = residual_block(u3, f[0])
    print(d3.shape)
    # u4 = upsample_concat_block(d3, e1)
    # d4 = residual_block(u4, f[1])
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d3)
    model = keras.models.Model(inputs, outputs)
    return model
