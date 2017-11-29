import numpy as np
import cv2
import sys
import os
import shutil
import timm

if os.path.exists('./data'):
	shutil.rmtree('./data') 
dirname = 'data'
os.mkdir(dirname)

face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('//usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_eye.xml')
# mouth_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')
mouth_cascade = cv2.CascadeClassifier('./haarcascade_mcs_mouth.xml')

# if len(sys.argv) < 2:
# 	sys.exit('Wrong Usage')

# image_name = sys.argv[1]

# img = cv2.imread(image_name)

camera = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('sample.avi',fourcc, 3, (1280,720))
counter = 0
kernel = np.ones((3,3),np.uint8)

while 1:
	retval, img = camera.read()
	# print img.shape
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# print 'y'
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		# print 'face found'
		cv2.rectangle(img, (x,y), (x+w, y+h), 0, 2)
		roi_face = gray[y:y+h, x:x+w]
		roi_face_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5)
		for (ex, ey, ew, eh) in eyes:
				counter += 1
				cv2.rectangle(roi_face_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
				# print "eye " + str(ex) + " " + str(ey)
				# roi_eye = roi_face[int(1.2*ey):int(0.8*(ey+eh)), int(1.2*ex):int(0.8*(ex+ew))]
				roi_eye = roi_face[ey:ey+eh, ex:ex+ew]
				center = 0
				
				roi_eye = cv2.GaussianBlur(roi_eye,(3,3),0)
				roi_eye = cv2.addWeighted(roi_eye,1.5,roi_eye,-0.5,0)
				roi_eye_canny = cv2.Canny(roi_eye,100,200)
				cv2.imwrite('./data/canny' + str(counter) + '.png', roi_eye_canny)
				laplacian = cv2.Laplacian(roi_eye,cv2.CV_64F)
				cv2.imwrite('./data/lapla' + str(counter) + '.png', laplacian)
				# res = cv2.resize(roi_eye,(int(ew/2), int(eh/2)), interpolation = cv2.INTER_AREA)
				roi_eyex = cv2.Sobel(roi_eye, cv2.CV_64F, 1, 0, ksize=3)
				roi_eyey = cv2.Sobel(roi_eye, cv2.CV_64F, 0, 1, ksize=3)
				roi_eyex = np.absolute(roi_eyex)
				roi_eyey = np.absolute(roi_eyey)
				roi_eyex = np.uint8(roi_eyex)
				roi_eyey = np.uint8(roi_eyey)
				# sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
				# abs_sobel64f = np.absolute(sobelx64f)
				# sobel_8u = np.uint8(abs_sobel64f)

				cv2.imwrite('./data/zsobely' + str(counter) + '.png', roi_eyey)
				cv2.imwrite('./data/zsobelx' + str(counter) + '.png', roi_eyex)
				ret, tmp = cv2.threshold(roi_eyex, 0, 255, cv2.THRESH_OTSU)
				tmp = cv2.erode(tmp, kernel, iterations=1)
				cv2.imwrite('./data/zsobelxt' + str(counter) + '.png', tmp)
				
				mag = np.hypot(roi_eyex, roi_eyey)  # magnitude
				mag *= 255.0 / np.max(mag)  # normalize (Q&D)
				roi_eye_sobel = mag.astype(np.uint8)
				# roi_eye_sobel = cv2.morphologyEx(roi_eye_sobel, cv2.MORPH_OPEN, kernel)
				cv2.imwrite('./data/xy' + str(counter) + '.png', roi_eye_sobel)
				# roi_eye_sobel = cv2.morphologyEx(roi_eye_sobel, cv2.MORPH_OPEN, kernel)
				# roi_eye_sobel = cv2.erode(roi_eye_sobel, kernel, iterations = 1)
				# roi_eye_sobel = cv2.morphologyEx(roi_eye_sobel, cv2.MORPH_CLOSE, kernel)
				ret, roi_eye_sobel = cv2.threshold(roi_eye_sobel, 0, 255, cv2.THRESH_OTSU)
				roi_eye_sobel = cv2.erode(roi_eye_sobel, kernel, iterations=1)
				cv2.imwrite('./data/tempthresh' + str(counter) + '.png', roi_eye_sobel)
				
				roi_eye_color = roi_face_color[ey:ey+eh, ex:ex+ew]	
				# center = timm.findEyeCenter(roi_eye_color, (0,0))
				# cv2.circle(roi_eye_color, center, 5, (255, 255, 255), 2)					
				
				pupils = cv2.HoughCircles(roi_eye_sobel, cv2.HOUGH_GRADIENT, 1, 100, param1 = 100, param2 = 10, minRadius=int(ew/11), maxRadius=int(ew/3))
				if pupils is not None:
					# print 'not none'
					pupils = np.round(pupils[0,:]).astype("int")
					for (x,y,r) in pupils:
						print str(x) + " " + str(y) + " " + str(r) + " --- " + str(counter) + " " + str(int(ew/11)) + "-" + str(int(ew/3))
						# cv2.circle(roi_eye_color, (x, y), r, (255, 165, 0), 2)
						cv2.circle(roi_eye_color, (x, y), 2, (255, 165, 0), 3)
						# cv2.imshow('eye' + str(x), roi_eye_color)
						# print roi_eye_sobel.shape
						# print roi_eye_color.shape
						comb = np.zeros(shape=(roi_eye_color.shape[0], roi_eye_color.shape[1]*2, roi_eye_color.shape[2]), dtype=np.uint8)

						comb[:roi_eye_color.shape[0], :roi_eye_color.shape[1]] = roi_eye_color
						comb[:roi_eye_sobel.shape[0], roi_eye_sobel.shape[1]:] = roi_eye_sobel[:, :, None]
						# cat = np.concatenate([roi_eye_sobel, roi_eye_color])
						cv2.imwrite('./data/eye' + str(counter) + '.png', comb)
						# cv2.moveWindow('eye' + str(x), 1000, 100)
						# cv2.resizeWindow('eye' + str(x), eh*2, ew*2)
		

		# mouths = mouth_cascade.detectMultiScale(roi_face, 1.7, 11)
		# for (mx, my, mw, mh) in mouths:
		# 	cv2.rectangle(roi_face_color, (mx, my), (mx+mw, my+mh), (0, 0, 0), 2)
		# 	roi_mouth = roi_face[my:my+mh, mx:mx+mw]
		# 	roi_mouth_color = roi_face_color[my:my+mh, mx:mx+mw]
		# 	roi_mouth = cv2.cornerHarris(roi_mouth, 2, 3, 0.04)
		# 	roi_mouth = cv2.dilate(roi_mouth, None)
		# 	roi_mouth_color[roi_mouth>0.01*roi_mouth.max()]=[0,0,255]
	out.write(img)	
	cv2.imshow('test', img)
	# cv2.imshow('bhawsar', gray)
	# cv2.moveWindow('bhawsar', 800,100)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break



camera.release()
out.release()

# cv2.waitKey(0)
cv2.destroyAllWindows()