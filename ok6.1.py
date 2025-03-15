#--------------------------
# Copyright (c) 2025 Boulbaba Guedri
# Licence : MIT
# Corresponding Author : boulbaba.guedri@ensit.u-tunis.tn (Boulbaba Guedri),naji.guedri@ensit.u-tunis.tn (Naji Guedri), rached.gharbi@ensit.rnu.tn (Rached Gharbi).
# Institutions : Laboratoire d’Ingénierie des Systèmes Industriels et des Energies Renouvelables (LISIER). Ecole Nationale Supérieure d'Ingénieurs de Tunis (ENSIT), University of Tunis, 05 Ave Taha Hussein, 1008 Montfleury Tunis, TUNISIA
# Smart Camera Two (SCT) : model SCT-O
# Journal : The visual computer
# Year : 2025
# Title of Manuscript: Advanced Real-time Angular Tracking of Human Arm Gestures Using a Smart Camera System Integrated with CNN and Fuzzy Logic
#--------------------------

from time import sleep
import datetime
import time
import string
import numpy as np
import cv2
import math
from math import sqrt
import numpy as np
import io
import os
import glob
import shutil
from numpy import array
import threading

np.float = float
np.int = int

liste_1 = []
liste_2 = []
liste_3 = []

#------------------

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variables shared between threads
frame = None
faces = []
running = True

def video_capture():
    global frame, running
    cap = cv2.VideoCapture("rtsp://admin:your_password@169.254.53.26:554/Streaming/Channels/102")
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        # Redimensionner l'image à 640x480-------
        resized_frame = cv2.resize(frame, (640, 480))

    cap.release()

def detect_faces():
    global frame, faces, running
    while running:
        if frame is not None:
            # Convert to grayscale and apply blur
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            #ret , th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
 
            # Perform detection
            faces = face_cascade.detectMultiScale(blur, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
# Start threads for video capture and detection
capture_thread = threading.Thread(target=video_capture)
detection_thread = threading.Thread(target=detect_faces)
capture_thread.start()
detection_thread.start()

# Show video stream
while True:

    #++++++++++++++++++++++++++++++ Read file contains HSV from calibration program, goal of this given for thresholding.
    
    valeur1 =open('C:/Users/najig/Desktop/Test_Camera/Annexe_Programmes_Livre/Parametres_HSV','r') # Call to a file containing thresholding parameters
    v1 = valeur1.readline ()
    v11=int(v1)
    v2 = valeur1.readline ()
    v22=int(v2)
    v3 = valeur1.readline ()
    v33=int(v3)
    v4 = valeur1.readline ()
    v44=int(v4)
    v5 = valeur1.readline ()
    v55=int(v5)
    v6 = valeur1.readline ()
    v66=int(v6)
    valeur1.close()
    lower_couleur = np.array([v11, v22, v33])
    upper_couleur = np.array([v44, v55, v66])

#-------------------------------------------------------------------- Determine the position of the arms
    if frame is not None:
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            xc = (x+(w/2))
            xcc = round(xc)
            yc = (y+(h/2))
            ycc = round(yc)
            ggi = xc-x
            ggio = round(ggi)
            ggii = yc-y
            ggiio = round(ggii)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            #print ("valeur y est :", y)


            nombre1 = (xc/3)*1
            x1 = round(nombre1)

            nombre2 = (xc/3)*2
            x2 = round(nombre2)

            nombre3 = (xc/3)*3
            x3 = round(nombre3)


            # Convert to grayscale and apply blur
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

#--------------------------------------------------------------------------------------------- threshold
    
            # Remarue : To threshold an image, there are two methods :
            # Method 1 : 
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            th = cv2.inRange(hsv, lower_couleur, upper_couleur)
            # Method 2 :
            #ret , th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            cv2.imwrite('C:/Users/najig/Desktop/Test_Camera/Annexe_Programmes_Livre/ImageBinaireAFM/Thresh-Before-AFM.jpg',th)

#------------------------------------------------------------------------------------------------AFM filter application

            
            # Apply an operation to eliminate morphological noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            cleaned = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
            cv2.imwrite('C:/Users/najig/Desktop/Test_Camera/Annexe_Programmes_Livre/ImageBinaireAFM/Thresh to Morphology.jpg',cleaned) 

            # Find the outlines of objects
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create an empty mask
            mask = np.zeros_like(cleaned)

            # Detecting contours: Find the largest contour (assumed to be the user's body)

            largest_contour = max(contours, key=cv2.contourArea)
            # Dessiner le plus grand contour sur le masque
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            # Appliquer le masque à l'image binaire
            result = cv2.bitwise_and(th, mask)
            cv2.imwrite('C:/Users/najig/Desktop/Test_Camera/Annexe_Programmes_Livre/ImageBinaireAFM/Thresh-After-AFM.jpg',result)

#---------------------------------------------------------------------------------------------------------------- Detect arm position
            # Determine Gain, RQ: I already talked about this Gain above
            aera_rec = w * h
            #print(aera_rec)

            Gain = (aera_rec/7500) * 50 

            myArray1 = np.array(result) # myArray1 = np.array(th)
            slice1 = myArray1[:480,x1]
            slice2 = myArray1[:480,x2]
            slice3 = myArray1[:480,x3]
            nn1 = np.where(slice1 == 255)
            g = nn1[0]
            #y1 = g[0]
            if len(g) > 0:
               y1 = g[0]
            else:
               print("")
            nn2 = np.where(slice2 == 255)
            g2 = nn2[0]
            #y2 = g2[0]
            if len(g2) > 0:
               y2 = g2[0]

               #print ("valeur y2 est :", y2)


               # Implement a variable angular scale that adjusts based on user movement in front of the camera.

               # Step 1) ---------- determine starting part (initial)

               yy3i = y - 0
               y3i = int (yy3i)
               cv2.putText(frame, f"-{135}C", (0, y3i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

               # Step 2)---------- determine the top part

               yy2i = y3i - Gain
               y2i = int (yy2i)
               cv2.putText(frame, f"-{120}C", (0, y2i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

               yy1i = y2i - Gain
               y1i = int (yy1i)
               cv2.putText(frame, f"-{90}C", (0, y1i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

               # Step 3)---------- determine the bottom part

               yy4i = y3i + Gain
               y4i = int (yy4i)  
               cv2.putText(frame, f"-{150}C", (0, y4i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

               yy5i = y4i + Gain
               y5i = int (yy5i)  
               cv2.putText(frame, f"-{165}C", (0, y5i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

               yy6i = y5i + Gain
               y6i = int (yy6i)
               cv2.putText(frame, f"-{180}C", (0, y6i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

               yy7i = y6i + Gain
               y7i = int (yy7i)
               cv2.putText(frame, f"-{195}C", (0, y7i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

               yy8i = y7i + Gain
               y8i = int (yy8i)
               cv2.putText(frame, f"-{210}C", (0, y8i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)               

               yy9i = y8i + Gain
               y9i = int (yy9i)
               cv2.putText(frame, f"-{225}C", (0, y9i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

               yy10i = y9i + Gain
               y10i = int (yy10i)  
               cv2.putText(frame, f"-{240}C", (0, y10i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


               # determine hand position
               
               if (y2 <= y1i):
                 #print ("90°C")
                 # Show classification on image
                 cv2.putText(frame, f"Angle: {90}C", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

               elif (y1i <= y2 <= y2i):
                 #print ("120°C")
                 cv2.putText(frame, f"Angle: {120}C", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

               elif (y2i <= y2 <= y3i):
                 #print ("135°C")
                 cv2.putText(frame, f"Angle: {135}C", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

               elif (y3i <= y2 <= y4i):
                 #print ("150°C")
                 cv2.putText(frame, f"Angle: {150}C", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

               elif (y4i <= y2 <= y5i):
                 #print ("165°C")
                 cv2.putText(frame, f"Angle: {165}C", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

               elif (y5i <= y2 <= y6i):
                 #print ("180°C")
                 cv2.putText(frame, f"Angle: {180}C", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #150

               elif (y6i <= y2 <= y7i):
                 #print ("195°C")
                 cv2.putText(frame, f"Angle: {195}C", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #165

               elif (y7i <= y2 <= y8i):
                 #print ("210°C")
                 cv2.putText(frame, f"Angle: {210}C", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

               elif (y8i <= y2 <= y9i):
                 #print ("225°C")
                 cv2.putText(frame, f"Angle: {225}C", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
               elif (y9i <= y2 <= y10i):
                 #print ("240°C")
                 cv2.putText(frame, f"Angle: {240}C", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


               elif (y10i <= y2 <= 480):
                 #print ("270°C")
                 cv2.putText(frame, f"Angle: {270}C", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            
               else:
                 #print("")
                 cv2.putText(frame, f"Angle:    ", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            nn3 = np.where(slice3 == 255)
            g3 = nn3[0]
            #y3 = g3[0]

            if len(g3) > 0:
               y3 = g3[0]
            else:
               print("")


            cv2.imshow("Flux en temps réel", frame)


    if cv2.waitKey(1) & 0xFF == ord("\r"): 
        break
        cap.close()

# Stop threads
capture_thread.join()
detection_thread.join()
cv2.destroyAllWindows()

