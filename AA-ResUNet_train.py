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
LEARNING_RATE = " Set the Learning Rate"

model_path = "Set the Directory of where you want to save model/"

from Model import AA_ResUNet
from IoU_Loss import iou
from Evaluation_Metrics import precision,recall,f1_score
from Data_Augmentation import train_generator,validation_generator


def train(model_name):
    K.clear_session()
    model = AA_ResUNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    model.summary()
    adam = keras.optimizers.Adam(LEARNING_RATE)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=["accuracy",precision,recall,f1_score, iou])
    
    t = time.time()
    #earlystopper = EarlyStopping(patience=3, verbose=1)
    checkpointer = ModelCheckpoint(model_path + "/model_" + model_name + "-{epoch:02d}-{loss:4f}_{accuracy:.4f}_{iou:.4f}_{dice_coef:.4f}_.hd5", monitor='val_loss', verbose=1, save_best_only=True)

    history = model.fit_generator(train_generator,validation_data=validation_generator,
                              validation_steps=130,steps_per_epoch=650,
                              epochs=200,callbacks=[checkpointer])
    print("Training Time = %s"%(t - time.time()))
    
if __name__ == "__main__":
    train(model_name="Resnet_att_aug")