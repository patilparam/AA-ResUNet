
#https://github.com/titu1994/keras-attention-augmented-convs/blob/master/attn_augconv.py

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from attn_augconv import augmented_conv2d
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D

ip = Input(shape=(256, 256, 3))
x = augmented_conv2d(ip, filters=20, kernel_size=(3, 3),strides = 1,
                     depth_k=4, depth_v=4,  # dk/v (0.2) * f_out (20) = 4
                     num_heads=2, relative_encodings=True)

model = Model(ip, x)
model.summary()

# Check if attention builds properly
x = tf.zeros((1, 32, 32, 3))
y = model(x)
print("Attention Augmented Conv out shape : ", y.shape)


ip = Input(shape=(32, 32, 3))
cnn1 = SeparableConv2D(filters = 1, kernel_size=3, strides=1,padding='same')(ip)
x = augmented_conv2d(cnn1, filters=20, kernel_size=3, # shape parameter is not needed
                     strides = 1,
                     depth_k=4, depth_v=4,  # padding is by default, same
                     num_heads=4, relative_encodings=True)

# depth_k | filters, depth_v | filters,  Nh | depth_k, Nh | filters-depth_v

model = Model(ip, x)
model.summary()