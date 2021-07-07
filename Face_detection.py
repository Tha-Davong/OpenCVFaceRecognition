import cv2 as cv
import numpy as np
import os

people = []
DIR = 'Resources/val/madonna'
har_cascade = cv.CascadeClassifier('face_detection.xml')


img_path = os.path.join(DIR, '1.jpg')
read_image = cv.imread(img_path)
gray = cv.cvtColor(read_image, cv.COLOR_BGR2GRAY)
face_recognized = har_cascade.detectMultiScale(gray, 1.1, 4)
for(x,y,w,h) in face_recognized:
    read_image = cv.rectangle(read_image, (x,y), (x+w, y+h), (255, 0 ,0), 1)
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', read_image.shape[1], read_image.shape[0])
    cv.imshow('image', read_image)
    cv.waitKey(0)








