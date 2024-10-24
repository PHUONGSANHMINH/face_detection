from cProfile import label
from pickletools import optimize

import numpy as np
import os
from PIL import Image

TRAIN_DATA = 'datasets/train-data'
TEST_DATA ='datasets/test_data'
Xtrain = []
ytrain = []

Xtest = []
ytest = []

dict = {'posBThy': [1, 0, 0, 0, 0], 'posTHoang': [0, 1, 0, 0, 0],
        'testBThy': [1, 0, 0, 0, 0], 'testTHoang': [0, 1, 0, 0, 0]}


def getData(dirData, lstData):
    for whatever in os.listdir(dirData):
        whatever_path = os.path.join(dirData, whatever)
        lst_filename_path = []
        for filename in os.listdir(whatever_path):
            filename_path = os.path.join(whatever_path, filename)
            label = filename_path.split('\\')[1]
            img = np.array(Image.open(filename_path))
            lst_filename_path.append((img, dict[label]))

        lstData.extend(lst_filename_path)
    return lstData
Xtrain = getData(TRAIN_DATA,Xtrain)
Xtest = getData(TEST_DATA,Xtest)

# print(Xtest[500])

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models
model_training_first = models.Sequential([
     layers.Conv2D(32,(3,3), input_Shape=(100,100,3), activation='relu'),
     layers.MaxPool2D((2,2)),
     layers.Dropout(0.2),

     layers.Conv2D(64,(3,3), activation='relu'),
     layers.MaxPool2D((2,2)),
     layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(1000,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(10,activation='softmax'),
 ])
model_training_first.summary()
# model_training_first.compile(optimize='adam',
#                              loss='categorycal_crossentropy',
#                              metrics=['accuracy'])
# model_training_first.fit(Xtrain,Ytrain_ohc,epochs=10)