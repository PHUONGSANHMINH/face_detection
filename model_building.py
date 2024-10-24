from cProfile import label

import numpy as np
import os
from PIL import Image

TRAIN_DATA = 'datasets/train-data'

Xtrain = []
ytrain = []

dict = {'posBThy': [1, 0, 0, 0, 0], 'posTHoang': [0, 1, 0, 0, 0]}

for whatever in os.listdir(TRAIN_DATA):
    whatever_path = os.path.join(TRAIN_DATA, whatever)
    lst_filename_path = []
    for filename in os.listdir(whatever_path):
        filename_path = os.path.join(whatever_path, filename)
        label = filename_path.split('\\')[1]
        img = np.array(Image.open(filename_path))
        lst_filename_path.append((img, dict[label]))

    Xtrain.extend(lst_filename_path)

    print(Xtrain)