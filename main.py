# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 19:53:52 2018

@author: shash
"""

# Imports
import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

#%% Load Model

from keras.models import load_model
model = load_model('Models/model.h5')
model2 = load_model('Models/model2.h5')

#%%

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

#%% Helper Functions
    
    # function to overlay a transparent image on background.
def transparentOverlay(src , overlay , pos=(0,0),scale = 1):
    overlay = cv2.resize(overlay,(0,0),fx=scale,fy=scale)
    h,w,_ = overlay.shape  # Size of foreground
    rows,cols,_ = src.shape  # Size of background Image
    y,x = pos[0],pos[1]    # Position of foreground/overlay image
    
    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(overlay[i][j][3]/255.0) # read the alpha channel 
            src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    return src    
    
def get_filter_spots(pred_res, scale_x, scale_y, scale, dog_filter):    
    #scale_x and y for scale of 96x96 image and scale for  overall filter scaling
    filter_nose = dog_filter['nose']
    filter_right_ear = dog_filter['ear_right']

    #Hyper-Paramaters
    y_padding = 5
    ear_padding = 6

    #Add nose
    nose_x = int(pred_res[20]*48+48*scale_x - filter_nose.shape[1]*scale/2)
    nose_y = int( (pred_res[21]*48+48 + y_padding)*scale_y - filter_nose.shape[0]*scale/2)
    
    #result = transparentOverlay(img_crop.copy(),filter_nose,(x,y),scale)
    
    left_ear_x = 0 - ear_padding
    left_ear_y = 0 - ear_padding*2
    
    right_ear_x = int( (96 + ear_padding*2)*scale_x - filter_right_ear.shape[0]*scale)
    right_ear_y = (0 - ear_padding)*scale_y
    
    return [nose_x, nose_y],[left_ear_x*scale_x, left_ear_y*scale_y],[right_ear_x, right_ear_y]

def get_best_scaling(w):
    filter_width = 420
    return 1.1*(w/filter_width)
    
def get_eye_angle(pred):
    left = pred[0:1]
    right = pred[2:3]
    #find vector
    vec = [right[0]-left[0], right[1]-left[1] ]
    angle = vec[0] / (math.sqrt(vec[0]*vec[0] + vec[1]*vec[1]))
    
    return angle
    
#%% Main Function to apply dog filter
def add_dog_filter(img_path, dog_filter):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)

    filter_nose = dog_filter['nose']
    filter_left_ear = dog_filter['ear_left']
    filter_right_ear = dog_filter['ear_right']
    
    if len(faces)==0:
        print("No faces Detected in the image")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for (x,y,w,h) in faces:        
        # add bounding box to color image
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        img_crop = img[y:y+h, x:x+w]
        
        scale_x = img_crop.shape[0]/96
        scale_y = img_crop.shape[1]/96
        
        img2 = cv2.resize(img_crop, (96,96))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        #for model
        img2 = img2.astype(np.float32)
        img2 = img2 / 255.  # scale pixel values to [0, 1]
        img2 = img2.reshape(-1, 96, 96, 1)
        
        #Predict using CNN model
        pred_res = model2.predict(img2)[0]

        scale = get_best_scaling(w)
        
        nose,left_ear,right_ear = get_filter_spots(pred_res, scale_x, scale_y, scale, dog_filter)  
        
        #Add images
        result = transparentOverlay(img.copy(),filter_nose,( int(nose[0]+x), int(nose[1]+y)), scale)
        result = transparentOverlay(result.copy(),filter_left_ear,( int(left_ear[0]+x), int(left_ear[1]+y)), scale)
        result = transparentOverlay(result.copy(),filter_right_ear,( int(right_ear[0]+x), int(right_ear[1]+y)), scale)
        
        img = result
    #Change to RGB
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    #and return
    return result
    

#%%  Main Program
        
#read filter image
filter_path = 'filter3.png'
filter_full = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED) #Read woth PNG transparency
    
dog_filter = {  'nose' : filter_full[302:390,147:300],
                'ear_left' : filter_full[55:195,0:160],
                'ear_right' : filter_full[55:190,255:420],
             }

img_path = 'Images/img5.jpg'
result = add_dog_filter(img_path, dog_filter)

plt.imshow(result)
plt.show()
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite('Images/Edited/'+img_path[7:] ,result)
