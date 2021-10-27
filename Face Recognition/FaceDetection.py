import cv2
import matplotlib.pyplot as plt
import sys




face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('Test2.jpg')


#img = plt.imread('Test.jpg')



print(img)

cv2.imshow('img',img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray,1.01,5)

cv2.imshow('',gray)



for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)

cv2.imshow('img',img)
cv2.waitKey()