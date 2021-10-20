
import pandas as pd
import numpy as np
import cv2
import os
import glob
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import Sequential, models
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

import matplotlib.pyplot as plt
import seaborn as sns



# load json and create model
json_file = open('Saved_Models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Saved_Models/model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


image_size = 150



im0 = cv2.imread('Test/0.png')

im1 = cv2.imread('Test/1.png')

im2 = cv2.imread('Test/2.png')

im3 = cv2.imread('Test/3.png')

Xs = []
Xs.append(im0)

Xs.append(im1)

Xs.append(im2)

Xs.append(im3)

Xs = np.array(Xs)



for i in range (0,len(Xs)) : 
    
    img = cv2.cvtColor(Xs[i], cv2.COLOR_BGR2GRAY)
    Xs[i]= cv2.resize(img,(image_size,image_size))
    cv2.imshow('',Xs[i])
    cv2.waitKey(0)

print(Xs[0])

Xs = Xs/255
print(len(Xs))
print(len(Xs[0]))
print(len(Xs[0][0][0]))
Xs.reshape(len(Xs), len(Xs[0]), len(Xs[0][0][0]), 1)


#ypred = loaded_model.predict(Xs)
#print(ypred)


