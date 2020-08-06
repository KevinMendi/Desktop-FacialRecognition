import cv2,os
import numpy as np
import sqlite3
import pickle
from PIL import Image

recognizer = cv2.face_LBPHFaceRecognizer.create();
recognizer.read("recognizer/trainingData.yml")
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataSet'

def getProfile(id):
    
    conn=sqlite3.connect("FaceDb.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor = conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile


#font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
cam = cv2.VideoCapture(0);
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret,img = cam.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,  minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE);
    for(x,y,w,h) in faces:
        
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        profile=getProfile(id)
        if(profile!=None):
            cv2.putText(img,str(profile[1]),(x,y+h+40), font, 2,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(img,str(profile[2]),(x,y+h+80), font, 2,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(img,str(profile[3]),(x,y+h+120), font, 2,(0,255,0),2,cv2.LINE_AA)
            #cv2.putText(img,str(profile[4]),(x,y+h+120), font, 2,(0,255,0),2,cv2.LINE_AA)
            
        #cv2.putText(cv2.fromarray(img),str(id),(x,y+h), font, 255);
        
    cv2.imshow("Face", img);
    if(cv2.waitKey(1)==ord('q')):
        break;
        
       
        

cam.release()
cv2.destroyAllWindows()
