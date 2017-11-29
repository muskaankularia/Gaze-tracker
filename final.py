import numpy as np
import cv2
import sys
from GeneralHough import test as test
import shutil
import os
import csv

counter = 0

def face_detect(img, j, i, face_csv):
    global counter

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    x, y, w, h, i_reye, j_reye, j_leye, i_leye, j_mouth, i_mouth, j_nose, i_nose, i_leye_box, j_leye_box, ewl, ehl, i_reye_box, j_reye_box, ewr, ehr = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

    for (x,y,w,h) in faces:
        # print 'got faces'
        mid = w/2
        print mid
        counter += 1
        cv2.rectangle(img, (x,y), (x+w, y+h), 0, 2)
        roi_face = gray[y:y+h, x:x+w]
        roi_face_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_face)
        for (ex, ey, ew, eh) in eyes:

                print 'eye dimensions ex = ' + str(ex) + ' ey = ' + str(ey) + ' ew = ' + str(ew) + ' eh = ' + str(eh)
                # print 'got eyes'
                
                if ex < mid:
                    i_leye_box = ex
                    j_leye_box = ey
                    ewl = ew
                    ehl = eh
                else:
                    i_reye_box = ex
                    j_reye_box = ey
                    ewr = ew
                    ehr = eh
                cv2.rectangle(roi_face_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

                roi_eye = roi_face[ey:ey+eh, ex:ex+ew]
                roi_eye = cv2.GaussianBlur(roi_eye,(3,3),0)

                # roi_eye_c = cv2.Canny(roi_eye, 100, 50)
                roi_eye = cv2.addWeighted(roi_eye,1.5,roi_eye,-0.5,0)
                # res = cv2.resize(roi_eye,(int(ew/2), int(eh/2)), interpolation = cv2.INTER_AREA)
                roi_eyex = cv2.Sobel(roi_eye, cv2.CV_64F, 1, 0, ksize=3)
                roi_eyey = cv2.Sobel(roi_eye, cv2.CV_64F, 0, 1, ksize=3)
                roi_eyex = np.absolute(roi_eyex)
                roi_eyey = np.absolute(roi_eyey)
                roi_eyex = np.uint8(roi_eyex)
                roi_eyey = np.uint8(roi_eyey)

                mag = np.hypot(roi_eyex, roi_eyey)  # magnitude
                mag *= 255.0 / np.max(mag)  # normalize (Q&D)
                roi_eye_sobel = mag.astype(np.uint8)
    
                # cv2.imwrite('./data/canny' + str(counter) + '.png', roi_eye_c)
                # cv2.imwrite('./data/sobelx' + str(counter) + '.png', roi_eyex)
                # cv2.imwrite('./data/sobely' + str(counter) + '.png', roi_eyey)
                # cv2.imwrite('./data/threshsobel' + str(counter) + '.png' , roi_eye_thresh)
                # cv2.imwrite('./data/xy' + str(counter) + '.png', roi_eyex+roi_eyey)
                # ret2,roi_eye = cv2.threshold(roi_eye,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # roi_eye = cv2.GaussianBlur(roi_eye,(5,5),0)
                cv2.imwrite('./data/eye_temp' + str(counter) + '.png', roi_eye)
                # print (roi_eyex+roi_eyey).astype(dtype=np.uint8)
                roi_eye_color = roi_face_color[ey:ey+eh, ex:ex+ew]

                pupils = cv2.HoughCircles(roi_eye_sobel, cv2.cv.CV_HOUGH_GRADIENT, 1, 100, param1 = 50, param2 = 30, minRadius=0, maxRadius=int(ew/2))
                if pupils is not None:
                    # print 'not none'
                    pupils = np.round(pupils[0,:]).astype("int")
                    for (x,y,r) in pupils:
                        # print str(x) + " " + str(y) + " " + str(r)
                        if ex < mid:
                            i_leye = x
                            j_leye = y
                        else:
                            i_reye = x
                            j_reye = y
                        cv2.circle(roi_eye_color, (x, y), r, (255, 165, 0), 1)
                        cv2.circle(roi_eye_color, (x, y), 1, (255, 165, 0), 1)

                        comb = np.zeros(shape=(roi_eye_color.shape[0], roi_eye_color.shape[1]*2, roi_eye_color.shape[2]), dtype=np.uint8)

                        comb[:roi_eye_color.shape[0], :roi_eye_color.shape[1]] = roi_eye_color
                        comb[:roi_eye_sobel.shape[0], roi_eye_sobel.shape[1]:] = roi_eye_sobel[:, :, None]
                        # cat = np.concatenate([roi_eye_sobel, roi_eye_color])
                        cv2.imwrite('./data/eye' + str(counter) + '.png', comb)
                else: print 'no pupils'
            
        
        mouths = mouth_cascade.detectMultiScale(roi_face, 1.7, 11)
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(roi_face_color, (mx, my), (mx+mw, my+mh), (0, 0, 0), 2)
            roi_mouth = roi_face[my:my+mh, mx:mx+mw]
            roi_mouth = cv2.GaussianBlur(roi_mouth,(5,5),0)
            roi_mouth_color = roi_face_color[my:my+mh, mx:mx+mw]

            # i_mouth, j_mouth 
            ret_m = test(roi_mouth, roi_face, './data_img/mouth' + str(counter) + str(x) + str(y) + '.png')

            if ret_m is not None:
                i_mouth, j_mouth = ret_m[0], ret_m[1]

            # roi_mouth = cv2.cornerHarris(roi_mouth, 2, 3, 0.04)
            # roi_mouth = cv2.dilate(roi_mouth, None)
            # roi_mouth_color[roi_mouth>0.01*roi_mouth.max()]=[0,0,255]

        noses = nose_cascade.detectMultiScale(roi_face, 1.7, 11)
        for (mx, my, mw, mh) in noses:
            cv2.rectangle(roi_face_color, (mx, my), (mx+mw, my+mh), (0, 0, 0), 2)
            roi_nose = roi_face[my:my+mh, mx:mx+mw]
            roi_nose = cv2.GaussianBlur(roi_nose,(5,5),0)
            roi_nose_color = roi_face_color[my:my+mh, mx:mx+mw]

            ret_n = test(roi_nose, roi_face, './data_img/nose' + str(counter) + str(x) + str(y) + '.png')
            if ret_n is not None:
                i_nose, j_nose = ret_n
            # roi_mouth = cv2.Canny(roi_mouth, 80, 80)
            # roi_mouth = cv2.cornerHarris(roi_mouth, 2, 3, 0.04)
            # roi_mouth = cv2.dilate(roi_mouth, None)
            # roi_mouth_color[roi_mouth>0.15*roi_mouth.max()]=[0,0,255]
            # cv2.imwrite('./data_img/mouth_corner' + str(counter) + str(x) + str(y) + '.png', roi_mouth_color)
        face_csv.writerow((j, i, x, y, w, h, i_reye, j_reye, j_leye, i_leye, j_mouth, i_mouth, j_nose, i_nose, i_leye_box, j_leye_box, ewl, ehl, i_reye_box, j_reye_box, ewr, ehr))
        cv2.imwrite('./data_img/fullface' + str(counter) + '.png', img)



if os.path.exists('./data_img'):
    shutil.rmtree('./data_img')
dirname = 'data_img'
os.mkdir(dirname)

face_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_mcs_mouth.xml")
nose_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_mcs_nose.xml")


camera = cv2.VideoCapture(0)

swidth = 1920
sheight = 1080


factor1 = swidth/2
factor2 = sheight/2

if os.path.exists('./face_data.csv'):
    fp = open('face_data.csv', 'a') 
    face_csv = csv.writer(fp)
else:
    fp = open('face_data.csv', 'a') 
    face_csv = csv.writer(fp)
    face_csv.writerow(['j', 'i', 'x', 'y', 'w', 'h', 'i_reye', 'j_reye', 'j_leye', 'i_leye', 'j_mouth', 'i_mouth', 'j_nose', 'i_nose', 'i_leye_box', 'j_leye_box', 'ewl', 'ehl', 'i_reye_box', 'j_reye_box', 'ewr', 'ehr'])

# face_csv = csv.writer(fp, fieldnames = ['j', 'i', 'x', 'y', 'w', 'h', 'i_reye', 'j_reye', 'j_leye', 'i_leye', 'j_mouth', 'i_mouth', 'j_nose', 'i_nose'])

cv2.namedWindow("window", cv2.WINDOW_NORMAL)
for i in range(3):
    for j in range(3):
        dotted_image = np.zeros((sheight, swidth), dtype=np.uint8)
        cv2.circle(dotted_image,(i*factor1,j*factor2),15,(255,255,0),-1)
  
        while 1:            
            retval, img = camera.read()
            k = cv2.waitKey(5)%256

            if k == 32:    
                face_detect(img, j, i, face_csv)
                print 'done'
                break
            else:
                cv2.imshow('test', img)
                cv2.imshow("window", dotted_image)

face_csv = csv.writer([])
camera.release()
cv2.destroyAllWindows()