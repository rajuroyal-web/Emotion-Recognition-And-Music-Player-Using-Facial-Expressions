from tensorflow.keras.utils import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
from tkinter import*
import playsound 
from gtts import gTTS 
import os
from matplotlib import pyplot as plt 


num=1
def speaks(output):
    global num
    num +=1

 
    tospeak = gTTS(text=output, lang='en-US', slow=False)
    file = str(num)+".mp3"
    tospeak.save(file)
    playsound.playsound(file, True)
    os.remove(file)

speaks("SONGS WILL BE PLAYED ACCORDING TO THE EMOTION DETECTED")

detection_model_path =  'Song-recommend-system-using-emotions-main\haarcascade_files\haarcascade_frontalface_default.xml'
emotion_model_path = 'Song-recommend-system-using-emotions-main\models\_mini_XCEPTION.102-0.66.hdf5'


face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]



cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
    else:
        import tkinter as tk
        from tkinter import messagebox
        root123= tk.Tk()
        root123.withdraw()
        msgbox=tk.messagebox.showinfo('ERROR MESSAGE', "FACE NOT DETECTED,PLEASE TRY AGAIN!!")
        break

 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)

             
                
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)


    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    # print(label)
    # while True:
    if cv2.waitKey(1) & 0xFF == ord('s'):
        variable=label
        
        # break
        if variable=="happy":
            speaks("EMOTION DETECTED IS HAPPY")
            
            import happy
            break
        elif variable=="sad":
            speaks("EMOTION DETECTED IS SAD")
           
            import sad
            break
        elif variable =="angry":
            speaks("EMOTION DETECTED IS ANGRY")
         
            import angry
            break
        elif variable =="scared":
            speaks("EMOTION DETECTED IS SCARED")
            
            import scared
            break
        elif variable =="surprised":
            speaks("EMOTION DETECTED IS SURPRISED")
           
            import surprised
            break
        elif variable =="neutral":
            speaks("EMOTION DETECTED IS NEUTRAL")
        
            import neutral
            break
        elif variable =="disgust":
            speaks("EMOTION DETECTED IS DISGUST")
           
            import disgust
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
