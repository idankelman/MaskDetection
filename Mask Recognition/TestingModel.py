
import pandas as pd
import numpy as np
import cv2
import os
import glob

import pathlib

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
json_file = open('Saved_Models/model_final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Saved_Models/model_final.h5")
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


'''
for i in range (0,len(Xs)) : 
    img = cv2.cvtColor(Xs[i], cv2.COLOR_BGR2GRAY)
    Xs[i]= cv2.resize(img,(image_size,image_size))
    cv2.imshow('',Xs[i])
    cv2.waitKey(0)
'''

D = []

#for i in range (4,6):
#chose_path = 'dataset\\12kData\Face Mask Dataset\Validation\WithoutMask'
chose_path = 'Test'
for path in pathlib.Path(chose_path).iterdir():
    info = path.stat()
    file_name, file_extention = os.path.splitext(path)    
    img_array = cv2.imread(file_name+file_extention, cv2.IMREAD_GRAYSCALE)

    ## Resizing the array
    new_image_array = cv2.resize(img_array, (image_size, image_size))

    ##Encoding the image with the label
    
    D.append(new_image_array)

    #cv2.imshow('',new_image_array)
    #cv2.waitKey(0)

print(D[0])

D = np.array(D)

D = D/255

print(D.shape)

D = D.reshape(len(D), D.shape[1], D.shape[2], 1)

print(D.shape)

ypred = loaded_model.predict(D)

score = loaded_model.evaluate(D, np.array([0]*len(D)), verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


print(ypred)


