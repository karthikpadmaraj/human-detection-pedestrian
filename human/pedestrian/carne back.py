#Based on Zed code - Person Fall detection using raspberry pi camera and opencv lib. Link: https://www.youtube.com/watch?v=eXMYZedp0Uo

import cv2
import time
import numpy as np


cap = cv2.VideoCapture('Dpedestria.mp4')
time.sleep(1)
ret, frame1 = cap.read()

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)
hsv[...,1] = 255

fgbg = cv2.createBackgroundSubtractorMOG2()
j = 0
while(1):
    ret, frame = cap.read()
    
    
    #Conver each frame to gray scale and subtract the background
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    fgmask = fgbg.apply(gray)
  
    cv2.imshow('input video',frame)

    #optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs,fgmask, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('optical flow estimation',rgb)
    
    #Find contours
    contours,hierchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    if contours:
        areas = []

        for contour in contours:
            ar = cv2.contourArea(contour)
            areas.append(ar)
        
        max_area = max(areas or [0])

        max_area_index = areas.index(max_area)

        cnt = contours[max_area_index]

        M = cv2.moments(cnt)
        
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.drawContours(fgmask, [cnt], 0, (255,255,255), 3, maxLevel = 0)
        
        if h < w:
            j += 1
            
        if j > 10:
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame, 'falling down', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255), 2)
            
            print ("FALL")

        if h > w:
            j = 0 
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, 'Nomral Human Behavior', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0), 2)
            


        cv2.imshow('output video', frame)
    
        if cv2.waitKey(33) == 27:
         break
cv2.destroyAllWindows()
