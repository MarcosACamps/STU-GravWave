# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:51:00 2024

@author: Marcos
"""

#%% IMPORT

import os

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from keras.initializers import orthogonal
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

P_MODELSAVE = 'saved_models'

os.makedirs(P_MODELSAVE, exist_ok=True)  # Creates folder if missing
P_LOGS = 'logs'
P_IMGSAVE = 'saved_images'

dirs = [P_MODELSAVE, P_LOGS, P_IMGSAVE]

saved_weight = os.path.join(P_MODELSAVE, 'dataweights.{epoch:02d}-{val_accuracy:.2f}.hdf5')

for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)

dataset_path = 'C:/Users/Marcos/ligoData/Practice_Directory'
batch_size = 20
epochs = 5

noise_factor = 1
inputShape = (256,256)

train_dir = 'C:/Users/Marcos/ligoData/Practice_Directory/train'
val_dir = 'C:/Users/Marcos/ligoData/Practice_Directory/val_dir'
test_dir = 'C:/Users/Marcos/ligoData/Practice_Directory/test'

train_data_clean= tf.keras.utils.image_dataset_from_directory(train_dir, labels = 'inferred')
val = tf.keras.utils.image_dataset_from_directory(val_dir, labels = 'inferred')
test_data_clean = tf.keras.utils.image_dataset_from_directory(test_dir, labels = 'inferred')
# (train_data_clean, _), (test_data_clean, _) = cifar10.load_data()

#%% SCALING INPUT DATA

def Conv2DLayer(x, filters, kernel, strides, padding, block_id, kernel_init=orthogonal()):
    prefix = f'block_{block_id}_'
    x = layers.Conv2D(filters, kernel_size=kernel, strides=strides, padding=padding,
                      kernel_initializer=kernel_init, name=prefix+'conv')(x)
    x = layers.LeakyReLU(name=prefix+'lrelu')(x)
    x = layers.Dropout(0.2, name=prefix+'drop')((x))
    x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
    return x

def Transpose_Conv2D(x, filters, kernel, strides, padding, block_id, kernel_init=orthogonal()):
    prefix = f'block_{block_id}_'
    x = layers.Conv2DTranspose(filters, kernel_size=kernel, strides=strides, padding=padding,
                               kernel_initializer=kernel_init, name=prefix+'de-conv')(x)
    x = layers.LeakyReLU(name=prefix+'lrelu')(x)
    x = layers.Dropout(0.2, name=prefix+'drop')((x))
    x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
    return x



def AutoEncdoer(inputShape):
    inputs = layers.Input(shape=inputShape)
    
    # 256 x 256
    conv1 = Conv2DLayer(inputs, 64, 3, strides=1, padding='same', block_id=1)
    conv2 = Conv2DLayer(conv1, 64, 3, strides=2, padding='same', block_id=2)
    
    # 128 x 128
    conv3 = Conv2DLayer(conv2, 128, 5, strides=2, padding='same', block_id=3)
    
    # 64 x 64
    conv4 = Conv2DLayer(conv3, 128, 3, strides=1, padding='same', block_id=4)
    conv5 = Conv2DLayer(conv4, 256, 5, strides=2, padding='same', block_id=5)
    
    # 32 x 32
    conv6 = Conv2DLayer(conv5, 512, 3, strides=2, padding='same', block_id=6)
    
    # 16 x 16
    deconv1 = Transpose_Conv2D(conv6, 512, 3, strides=2, padding='same', block_id=7)
    
    # 32 x 32
    skip1 = layers.concatenate([deconv1, conv5], name='skip1')
    conv7 = Conv2DLayer(skip1, 256, 3, strides=1, padding='same', block_id=8)
    deconv2 = Transpose_Conv2D(conv7, 128, 3, strides=2, padding='same', block_id=9)
    
    # 64 x 64
    skip2 = layers.concatenate([deconv2, conv3], name='skip2')
    conv8 = Conv2DLayer(skip2, 128, 5, strides=1, padding='same', block_id=10)
    deconv3 = Transpose_Conv2D(conv8, 64, 3, strides=2, padding='same', block_id=11)
    
    # 128 x 128
    skip3 = layers.concatenate([deconv3, conv2], name='skip3')
    conv9 = Conv2DLayer(skip3, 64, 5, strides=1, padding='same', block_id=12)
    deconv4 = Transpose_Conv2D(conv9, 64, 3, strides=2, padding='same', block_id=13)
    
    # 256 x 256
    skip3 = layers.concatenate([deconv4, conv1])
    conv10 = layers.Conv2D(3, 3, strides=1, padding='same', activation='sigmoid',
                       kernel_initializer=orthogonal(), name='final_conv')(skip3)

    
    return models.Model(inputs=inputs, outputs=conv10)



data_gen_args = dict(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.5, 1.2],
    shear_range=0.01,
    horizontal_flip=True,
    rescale=1/255,
    fill_mode='reflect',
    data_format='channels_last')

data_flow_args = dict(
    target_size=inputShape,
    batch_size=batch_size,
    class_mode='input') # Since we want to reconstruct the input
train_datagen = ImageDataGenerator(**data_gen_args)
val_datagen = ImageDataGenerator(**data_gen_args)

train_batches = train_datagen.flow_from_directory(
    dataset_path + '/train',
    **data_flow_args)

val_batches = val_datagen.flow_from_directory(
    dataset_path + '/train',
    **data_flow_args)

from keras.optimizers import SGD, Adam

model = AutoEncdoer((*inputShape, 3))
# model_opt = SGD(lr=0.005, decay=1-0.995, momentum=0.7, nesterov=False)
model_opt = Adam(learning_rate=0.002)

model.compile(optimizer=model_opt, loss='mse', metrics=['accuracy'])



modelchk = keras.callbacks.ModelCheckpoint(saved_weight, 
                                      monitor='val_accuracy', 
                                      verbose=1,
                                      save_best_only=True, 
                                      save_weights_only=False,
                                      mode='auto',
                                      save_=2)

tensorboard = keras.callbacks.TensorBoard(log_dir=P_LOGS,
                                          histogram_freq=0,
                                          write_graph=True,
                                          write_images=True)

csv_logger = keras.callbacks.CSVLogger(f'{P_LOGS}/keras_log.csv',
                                       append=True)
model.fit(train_batches,
                    steps_per_epoch = train_batches.samples // batch_size,
                    epochs=epochs,
                    verbose=1, 
                    validation_data=val_batches,
                    validation_steps = train_batches.samples // batch_size,
                    callbacks=[modelchk, tensorboard, csv_logger],
                    use_multiprocessing=False)

#%% eval

directoryModel= 'C:/Users/Marcos/.spyder-py3/saved_models'
model = keras.models.load_model(os.path.join(P_MODELSAVE,
                                            'phase2_weights.01-0.86.hdf5'))
test_datagen = ImageDataGenerator(**data_gen_args)

test_batches = test_datagen.flow_from_directory(
    dataset_path + '/test',
    **data_flow_args)

X,y = test_batches

score = model.evaluate(X, y, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

decoded_imgs = model.predict(X)

import matplotlib.pyplot as plt

n = 5

plt.figure(figsize=(40, 15))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X[i])
    ax.axis('off')

    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i])
    ax.axis('off')

plt.show()