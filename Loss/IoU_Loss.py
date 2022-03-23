# -*- coding: utf-8 -*-
import tensorflow as tf
from keras import backend as K
import numpy as np 

def iou(y_true, y_pred, smooth = 1):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    #sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    iou_acc = (intersection + smooth) / (union + smooth)
    return iou_acc

def dice_coef(y_true, y_pred):
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    smooth = 1.0
    return (2.0*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)