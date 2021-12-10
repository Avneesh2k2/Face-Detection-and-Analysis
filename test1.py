import numpy as np
import cv2
from deepface import DeepFace
face = r"C:\Users\Avneesh\Desktop\Project\haarcascade_frontalface_default.xml"
face_obj = cv2.CascadeClassifier(face)
cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_obj.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)

    for x,y,w,h in faces:
        gray = cv2.rectangle(gray, (x,y) , (x+w,y+h), (0,255,0),3)
        

         
    cv2.imshow('video',gray)

    try: 

        res = DeepFace.analyze(img,actions=['race'])
        
        print(res)
        
    except:
        print("Face Error !!")

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()