import cv2
import numpy as np
import sqlite3
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);

def insertOrUpdate(Id, Name):
    conn=sqlite3.connect("FaceDb.db")
    cmd="SELECT * FROM People WHERE ID="+str(Id)
    cursor = conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist == 1):
        cmd="UPDATE People SET Name"+str(Name)+"WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO People (ID,Name) VALUES ("+str(Id)+","+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()
        

id = input('enter user id: ')
name=input('enter your name: ')
insertOrUpdate(id, name)
sampleNum = 0;
while(True):
    ret,img = cam.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = faceDetect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,  minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE);
    for(x,y,w,h) in faces:
        sampleNum = sampleNum+1;
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x-50,y-50), (x+w,y+h+50), (0,255,0), 2)
        cv2.waitKey(100);
    cv2.imshow("Face", img);
    cv2.waitKey(100);
    if(sampleNum>20):
        cam.release()
        cv2.destroyAllWindows()
        
        break;
        


