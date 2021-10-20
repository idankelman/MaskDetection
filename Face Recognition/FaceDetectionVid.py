import cv2
import matplotlib.pyplot as plt
import sys

face_classifier = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
cap = cv2.VideoCapture(0)

while True:
    
    _,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.1,4)
   
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img',img)
    k= cv2.waitKey(30) & 0xff
    if k ==27:
        break

cap.release()