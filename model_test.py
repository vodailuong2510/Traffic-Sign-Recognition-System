# -*- coding: utf-8 -*-
"""test

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1993_XryvBvh2fTah8_qSSSbZJ-Y6FZdv
"""

import tensorflow as tf
from google.colab import drive
drive.mount("/content/drive")

cd/content/drive/MyDrive/Lượng và hành trình nghiêm cứu khoa học

link = "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"

!pip install wget

import wget
wget.download(link)

data ="./"
!unzip -q traffic-signs-data.zip -d $data

train_link=data +"train.p"
valid_link=data+"valid.p"
test_link=data+"test.p"

import pickle
with open(train_link, mode="rb") as f:
  train=pickle.load(f)
with open(valid_link, mode="rb") as f:
  valid=pickle.load(f)
with open(test_link, mode="rb") as f:
  test=pickle.load(f)

trainX=train["features"]
trainY=train["labels"]
validX=valid["features"]
validY=valid["labels"]
testX=test["features"]
testY=test["labels"]

import matplotlib.pyplot as plt

classNames = {0: 'Speed limit (20km/h)',
 1: 'Speed limit (30km/h)',
 2: 'Speed limit (50km/h)',
 3: 'Speed limit (60km/h)',
 4: 'Speed limit (70km/h)',
 5: 'Speed limit (80km/h)',
 6: 'End of speed limit (80km/h)',
 7: 'Speed limit (100km/h)',
 8: 'Speed limit (120km/h)',
 9: 'No passing',
 10: 'No passing for vehicles over 3.5 metric tons',
 11: 'Right-of-way at the next intersection',
 12: 'Priority road',
 13: 'Yield',
 14: 'Stop',
 15: 'No vehicles',
 16: 'Vehicles over 3.5 metric tons prohibited',
 17: 'No entry',
 18: 'General caution',
 19: 'Dangerous curve to the left',
 20: 'Dangerous curve to the right',
 21: 'Double curve',
 22: 'Bumpy road',
 23: 'Slippery road',
 24: 'Road narrows on the right',
 25: 'Road work',
 26: 'Traffic signals',
 27: 'Pedestrians',
 28: 'Children crossing',
 29: 'Bicycles crossing',
 30: 'Beware of ice/snow',
 31: 'Wild animals crossing',
 32: 'End of all speed and passing limits',
 33: 'Turn right ahead',
 34: 'Turn left ahead',
 35: 'Ahead only',
 36: 'Go straight or right',
 37: 'Go straight or left',
 38: 'Keep right',
 39: 'Keep left',
 40: 'Roundabout mandatory',
 41: 'End of no passing',
 42: 'End of no passing by vehicles over 3.5 metric tons'}

from sklearn.utils import shuffle

trainX, trainY=shuffle(trainX, trainY)

trainX=trainX.astype("float")/255.0
validX=validX.astype("float")/255.0
testX=testX.astype("float")/255.0

from sklearn.preprocessing import LabelBinarizer

lb=LabelBinarizer()

trainY=lb.fit_transform(trainY)
validY=lb.fit_transform(validY)

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model=Sequential()
width=32
height=32
classes=43
shape=(width, height, 3)

model.add(Conv2D(32, (3, 3), padding="same", input_shape=shape))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(Dense(classes))

model.add(Activation("softmax"))

aug = ImageDataGenerator(rotation_range=0.18, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

learning_rate = 0.01

epochs = 10
batch_size = 64

opt = SGD(learning_rate=learning_rate, momentum=0.9)

model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

print("Start training")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size), validation_data=(validX, validY), steps_per_epoch=trainX.shape[0]//batch_size, epochs=epochs, verbose=1)

model.save("luongvagiaothong.h5")

saved_model = tf.keras.models.load_model("luongvagiaothong.h5")

result = saved_model.predict(testX[1:2])

import numpy as np

final_result = np.argmax(result)

final_result = classNames[final_result]

final_result

plt.imshow(test["features"][1])
