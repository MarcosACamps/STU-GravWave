# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:22:15 2023

@author: Marcos
"""
# %% Import Modules

import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dropout, MaxPooling2D, Conv2D, Normalization, Dense, Flatten, ThresholdedReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#
import sklearn
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns

# %% Grab Dataset
train_dir = 'C:/Users/Marcos/ligoData/Practice_Directory/train'
val_dir = 'C:/Users/Marcos/ligoData/Practice_Directory/val_dir'
test_dir = 'C:/Users/Marcos/ligoData/Practice_Directory/test'

train = tf.keras.utils.image_dataset_from_directory(train_dir, labels = 'inferred')
val = tf.keras.utils.image_dataset_from_directory(val_dir, labels = 'inferred')
test = tf.keras.utils.image_dataset_from_directory(test_dir, labels = 'inferred')

print(test)
class_names = test.class_names
print(class_names)
# %%% check
plt.figure(figsize=(20, 20))
for images, labels in train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
# %% MODEL
img_height = 256
img_width = 256

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])


model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    layers.Resizing(128, 128),
    ThresholdedReLU(2.0),
    Conv2D(8, 5, activation = 'relu'),
    Conv2D(8, 5, activation = 'relu'),
    # Conv2D(8, 5, activation = 'relu'),
    MaxPooling2D(2, 2, "same"),
    Conv2D(16, 3, activation='relu'),
    Conv2D(16, 3, activation='relu'),
    Conv2D(16, 3, activation='relu'),
    Conv2D(16, 3, activation='relu'),
    MaxPooling2D(2, 2, "same"),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(2, 2, "same"),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(2, 2, "same"),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.BinaryCrossentropy(),
              metrics=['BinaryAccuracy'])
model.summary()

# %% TRAIN
epochs = 50

es = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
RLr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                        patience=3, verbose=1, min_delta=1e-3, min_lr=5e-5)
#es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(
    train,
    validation_data=val,
    epochs=epochs,
    callbacks = RLr
)

# %% EVALUATION
acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

fig = plt.figure()
plt.plot(acc, color='teal', label='accuracy')
plt.plot(val_acc, color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.ylim([0.0, 1.0])
plt.xlim([0, epochs])
plt.show()

fig = plt.figure()
plt.plot(loss, color='teal', label='loss')
plt.plot(val_loss,
         color='orange',
         label='val_loss')
plt.grid(visible='true')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.xlabel('Epochs')
plt.ylim([0.0, 1.0])
plt.xlim([0, epochs])
plt.show()

# %% Confusion Matrix (CURRENTLY NOT FUNCTIONAL)

# #model predictions
# y_pred = []  # store predicted labels
# y_true = []  # store true labels

# # iterate over the dataset
# for image_batch, label_batch in test:   # use dataset.unbatch() with repeat
#    # append true labels
#    y_true.append(label_batch)
#    # compute predictions
#    preds = model.predict(image_batch)
#    # append predicted labels
#    y_pred.append(np.argmax(preds, axis = - 1))

# # convert the true and predicted labels into tensors
# correct_labels = tf.concat([item for item in y_true], axis = 0)
# predicted_labels = tf.concat([item for item in y_pred], axis = 0)

# print('Confusion Matrix')
# print(confusion_matrix(test.classes, y_pred))