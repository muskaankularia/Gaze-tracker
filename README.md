# Gaze-tracker
tracking where pupils are focussing on the screen using webcam


final.py
-used for calibration to store the position on the screen and corresponding puipl position, face position, eye        position, mouth position, nose position.

GeneralHough.py 
-Reference: https://github.com/vmonaco/general-hough

linerTest.py 
-used for training the data so that when looking at the screen we can get the corresponding point on screen.

face_data.csv 
-Result from final.py is stored in this file

face_pic.py 
-track your pupil using webcam

**change the haar_cascades extension in files final.py and face_pic.py accordingly. 


WORK YET TO BE DONE:
incorprating the reults from ml
Since this small dataset couldnt give good results( 250 entries - 53% test accuracy )

