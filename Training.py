import cv2 as cv
import numpy as np
import os

people = []
DIR = 'Resources/train'
for person in os.listdir(DIR):
    people.append(person)
print(people)
har_cascade = cv.CascadeClassifier('face_detection.xml')

features = []
names = []


def face_training():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            read_image = cv.imread(img_path)
            if read_image is None:
                continue
            gray = cv.cvtColor(read_image, cv.COLOR_BGR2GRAY)
            face_recognized = har_cascade.detectMultiScale(gray, 1.1, 4)
            for(x,y,w,h) in face_recognized:
                face_croped = gray[y:y+h, x:x+h]
                read_image = cv.rectangle(read_image, (x,y), (x+w, y+h), (255, 0 ,0), 1)
                #cv.imwrite('Resources/' + img + 'anotated.jpg', read_image)
                #cv.imshow('image', read_image)
                #cv.waitKey(0)
                features.append(face_croped)
                names.append(label)


face_training()
print("Traning done")

features = np.array(features, dtype='object')
names = np.array(names)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features,names)
face_recognizer.save('traindata.yml')

