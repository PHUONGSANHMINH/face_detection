#  from cProfile import label
# from pickletools import optimize

import numpy as np
import os
from PIL import Image

# TRAIN_DATA = 'datasets/train-data'
# TEST_DATA ='datasets/test_data'
# Xtrain = []
# ytrain = []
#
# # Xtrain = [(matranhinh1, ohe1), (matranhinh2, ohe2), .........., (matranhinh1518, ohe1518)]
#
# # Xtrain[0][0], Xtrain[0][1]
#
# # Xtrain = [x[0] for i, x in enumerate(Xtrain)]
#
# Xtest = []
# ytest = []
#
# dict = {'posBThy': [1, 0, 0, 0, 0], 'posTHoang': [0, 1, 0, 0, 0],
#         'testBThy': [1, 0, 0, 0, 0], 'testTHoang': [0, 1, 0, 0, 0]}
#
#
# def getData(dirData, lstData):
#     for whatever in os.listdir(dirData):
#         whatever_path = os.path.join(dirData, whatever)
#         lst_filename_path = []
#         for filename in os.listdir(whatever_path):
#             filename_path = os.path.join(whatever_path, filename)
#             label = filename_path.split('\\')[1]
#             img = np.array(Image.open(filename_path))
#             lst_filename_path.append((img, dict[label]))
#
#         lstData.extend(lst_filename_path)
#     return lstData
# Xtrain = getData(TRAIN_DATA,Xtrain)
# Xtest = getData(TEST_DATA,Xtest)
#
# np.random.shuffle(Xtrain)
# np.random.shuffle(Xtrain)
# np.random.shuffle(Xtrain)
# # print(Xtrain[1500])
#
#
#
import tensorflow
from numpy.ma.core import reshape
from tensorflow.keras import layers
from tensorflow.keras import models
# model_training_first = models.Sequential([
#      layers.Conv2D(32,(3,3), input_Shape=(128,128,3), activation='relu'),
#      layers.MaxPool2D((2,2)),
#      layers.Dropout(0.2),
#
#      layers.Conv2D(64,(3,3), activation='relu'),
#      layers.MaxPool2D((2,2)),
#      layers.Dropout(0.2),
#
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPool2D((2, 2)),
#     layers.Dropout(0.2),
#
#     layers.Flatten(),
#     layers.Dense(1000,activation='relu'),
#     layers.Dense(256,activation='relu'),
#     layers.Dense(5,activation='softmax'),
#  ])
# # model_training_first.summary()
# model_training_first.compile(optimize='adam',
#                              loss='categorycal_crossentropy',
#                              metrics=['accuracy'])
# model_training_first.fit(np.array([x[0] for _, x in enumerate(Xtrain)]), np.array([y[1] for _, y in enumerate(Xtrain)]), epochs=10)
#
# model_training_first.save('model-family_10epochs.h5')

import cv2

lstResult = ['Bich Thy', 'Tan Hoang']

face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
models = models.load_model('model-family_10epochs.h5')

cam = cv2.VideoCapture('review.mp4')

while True:
   OK, frame= cam.read()
   faces = face_detector.detectMultiScale(frame,1.3,5 )

   for (x,y,w,h) in faces:
       roi = cv2.resize(frame[y: y+h, x: x+w], (128, 128))
       result = np.argmax(models.predict(roi,reshape((-1, 128, 128, 3))))
       cv2.rectangle(frame, (x, y), (x+w, y+h), (128,255,50), 1)
       cv2.putText(frame, lstResult[result], (x+15, y-15), cv2.FONT_ITALIC, 0.8, (255, 25, 255), 2)


       cv2.imshow('FRAME',frame)
       if cv2.waitKey(1) & 0xff == ord('q'):
           break


cam.release()

cv2.destroyAllWindows()
