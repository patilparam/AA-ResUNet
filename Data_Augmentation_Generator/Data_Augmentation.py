# -*- coding: utf-8 -*-

from keras.preprocessing import image
# Creating the training Image and Mask generator

train_images_path = "Training iamges PATH/"
train_masks_path =  "Training Masks PATH/"
valid_images_path = "Validation Images PATH/"
valid_masks_path = "Validation Masks PATH"

# test_images_path = "Test Images PATH"
# test_masks_path = "Test Mask PATH"
BATCH_SIZE = "Set the Batch Size as per your dataset"

mask_data_gen_args = dict(
                     #featurewise_center=False,
                     #samplewise_center=False,
                     #featurewise_std_normalization=False,
                     #samplewise_std_normalization=False,
                     #zca_whitening=True,
                     #zca_epsilon=1e-06,
                     #rotation_range=20,
                     #width_shift_range=0.0,
                     #height_shift_range=0.0,
                     #brightness_range=None,
                     #shear_range=0.0,
                     #zoom_range=0.15,
                     #fill_mode="constant",
                     #cval=0.0,
                     horizontal_flip=True,
                     vertical_flip=True,
                     rescale=1./255,
                     #validation_split=0.1,
                     )



train_data_gen_args =dict(
                          #featurewise_center=False,
                          #samplewise_center=False,
                          #featurewise_std_normalization=False,
                          #samplewise_std_normalization=False,
                          #zca_whitening=True,
                          #zca_epsilon=1e-06,
                          #rotation_range=20,
                          #width_shift_range=0.0,
                          #height_shift_range=0.0,
                          #brightness_range=None,
                          #shear_range=0.0,
                          #zoom_range=0.15,
                          #fill_mode="constant",
                          #cval=0.0,
                          horizontal_flip=True,
                          vertical_flip=True,
                          #rescale=1./255,
                          #validation_split=0.1,
                          )

mask_datagen = image.ImageDataGenerator(**mask_data_gen_args)
train_datagen = image.ImageDataGenerator(**train_data_gen_args)
 
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1

train_image_generator = train_datagen.flow_from_directory(train_images_path,
                                                          color_mode="rgb",
                                                          batch_size=BATCH_SIZE,
                                                          class_mode=None,seed=seed)


train_mask_generator = mask_datagen.flow_from_directory(train_masks_path,
                                                         color_mode="grayscale",
                                                         batch_size=BATCH_SIZE,
                                                         class_mode=None,seed=seed)

validation_image_generator = train_datagen.flow_from_directory(valid_images_path,
                                                               color_mode="rgb",
                                                               batch_size=BATCH_SIZE,
                                                               class_mode=None,seed=seed)



validation_mask_generator = mask_datagen.flow_from_directory(valid_masks_path,
                                                              color_mode="grayscale",
                                                              batch_size=BATCH_SIZE,
                                                              class_mode=None,seed=seed)


##creating a training and validation generator that generate masks and images
train_generator = (pair for pair in zip(train_image_generator, train_mask_generator))
validation_generator = (pair for pair in zip(validation_image_generator, validation_mask_generator))
