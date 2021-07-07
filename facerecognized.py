import os

import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('face_detection.xml')

people = []
DIR = 'Resources/val'
for person in os.listdir(DIR):
    people.append(person)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('traindata.yml')

img = cv.imread('Resources/val/elton_john/3.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
face_detected = haar_cascade.detectMultiScale(gray, 1.1, 1)

for (x,y,w,h) in face_detected:
    face_cropped = gray[y:y+h, x:x+h]
    cv.imshow('cropped', face_cropped)
    cv.waitKey(0)
    label, confidence = face_recognizer.predict(face_cropped)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    cv.putText(img, str(people[label]) + ' ' + str(confidence), (x,y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.namedWindow('detect face', cv.WINDOW_NORMAL)
cv.resizeWindow('detect face', img.shape[1], img.shape[0])
#img = cv.resize(img, (1920,1080))
cv.imshow('detect face', img)
cv.waitKey(0)
