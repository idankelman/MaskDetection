

#=================================================================================
#
#                                   Mask Detection System 
#
#=================================================================================




#=================================================================================
#                                   crucial libs imports
#=================================================================================


import numpy as np
import cv2
import random
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import cv2
import os
import glob
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin/bin")
import pathlib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import Sequential, models
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

import seaborn as sns







#=================================================================================
#                                   Functions
#=================================================================================




#=================================================================================
#                                   Load Functions
#=================================================================================


def loadModel(path):


    print("Opening Model From Disk ") 

    json_file = open(path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    print("Loaded model from disk")

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("Saved_Models/model_final.h5")
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return loaded_model



#=================================================================================
#                                   image processing
#=================================================================================

def getPrediction(gray,loaded_model):
    with tf.device('/gpu:0'):
        cropped_image = gray[y-feather:y+h+feather,x-feather:x+w+feather]
        cropped_image = cv2.resize(cropped_image, (image_size, image_size))


        D = []
        D.append(cropped_image)
        D = np.array(D)
        D = D/255
        D = D.reshape(len(D), D.shape[1], D.shape[2], 1)
        
        ypred = loaded_model.predict(D)

    return ypred



def SwitchCap(CapVideo,path):
    if CapVideo:
        try:
            cap= cv2.VideoCapture(0)
            print('Swtiching Source to Cam')
        except:
            print('NO CAMERA')
            cap =cv2.VideoCapture(path)

    else:
        print('Swtiching Source to Video')
        cap =cv2.VideoCapture(path)
    return cap



#=================================================================================
#                                    Load Files
#=================================================================================
#tf.debugging.set_log_device_placement(True)


model_path = 'Saved_Models/model_final.json'
face_c_path = 'haarcascade_frontalface_default.xml'
mouth_c_path = 'Mouth.xml'


face_cascade = cv2.CascadeClassifier(face_c_path)
mouth_cascade = cv2.CascadeClassifier(mouth_c_path)
loaded_model = loadModel(model_path)



# load json and create model

#=================================================================================
#                                    Control Variables 
#=================================================================================


# start frame/FPS
count = 0
FPs = 15


#image anlyzation controls
threshHold = 0.8
image_size = 150
feather = 0

#visualization controls
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
T_Font = 0.5
org = (30, 30)
Mask_C = (40, 240, 0)
NMask_C = (40, 0, 240)
thickness = 0
CapVideo = True
VideoPath = 'Video_test.mp4'


#print sentances 
Mask_T = "Thank You for wearing MASK"
NMask_T = "Please wear MASK to defeat Corona"



#starting the video capture
cap = cv2.VideoCapture(VideoPath)
#cap = cv2.VideoCapture(0)



#=================================================================================
#                                    anlyze video 
#=================================================================================


while True:

    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    count+=FPs

    ret, img = cap.read()
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw_threshold = 60
    

   
    #Hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    #Hist = Hist/256
    #print(bw_threshold)
    #plt.hist(gray.ravel(),256,[0,256])
    #plt.show()
    adp_th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 5, 1.8)
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)
    faces_adpt = face_cascade.detectMultiScale(adp_th, 1.1, 4)

    
    cv2.putText(img, "FPS{}".format(FPs),(0,30), font, font_scale, Mask_C, thickness, cv2.LINE_AA)
    
    if(len(faces) == 0 and len(faces_bw) == 0) and len(faces_adpt) == 0:
        cv2.putText(img, "No face found...", org, font, font_scale, Mask_C, thickness, cv2.LINE_AA)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)


            
            #print(D.shape)
            ypred = getPrediction(gray,loaded_model)
            

            org = (x,y)
            txt_org = (int(x),int(y-font_scale))
            YPred_T = 'predict : {}%'.format(round(ypred[0][0]*100,2))
            NPred_T = 'predict : {}%'.format(100-round(ypred[0][0]*100,2))


            if(ypred[0][0]>= threshHold):
                cv2.rectangle(img, org, (x + w, y + h), Mask_C, 2)
                cv2.putText(img, YPred_T, txt_org, font, T_Font, Mask_C, thickness, cv2.LINE_AA)
                #cv2.putText(img, Mask_T, org, font, font_scale, Mask_C, thickness, cv2.LINE_AA)
                #cv2.putText(img,'predict : {}%'.format(ypred*100),(x+ (w/2) ,y-font_scale),font,font_scale,Mask_C,thickness,cv2.LINE_AA)
            else:
                cv2.putText(img, NPred_T, txt_org, font, T_Font, NMask_C, thickness, cv2.LINE_AA)
                #cv2.putText(img,Pred_T,(x+ (w/2) ,y-font_scale),font,font_scale,NMask_C,thickness,cv2.LINE_AA)
                cv2.rectangle(img, org, (x + w, y + h), NMask_C, 2)
                #cv2.putText(img, NMask_T, org, font, font_scale, NMask_C, thickness, cv2.LINE_AA)
        
            

        
    cv2.imshow('img',img)
    #cv2.imshow('img',adp_th)
    k= cv2.waitKey(30) & 0xff
    #print(k)
    if k== 97: 
        FPs-=1
        if FPs<=0 :
            Fps = 1

    elif k==100:
        FPs+=1
        if FPs>30:
            Fps = 30 
            
    elif k ==27:
        i=2
        for (x, y, w, h) in faces: 
            cropped_image = img[y-feather:y+h+feather,x-feather:x+w+feather]
            cv2.imwrite('{}.png'.format(i),cropped_image)
            i+=1
        break

    elif k == 32:
        cap.release()
        cap =SwitchCap(CapVideo,VideoPath) 
        CapVideo = not CapVideo
        
   
cap.release()




