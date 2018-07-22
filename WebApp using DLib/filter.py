#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 22:49:58 2018

@author: shashwat
"""

# Imports
import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import dlib

def init_model():
    #%% Load Model
    from keras.models import load_model
    #model = load_model('model.h5')
    model2 = load_model('model2.h5')
    return model2
#%%
def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

#%%  function to overlay a transparent image on background.
def transparentOverlay(src , overlay , pos=(0,0),scale = 1, angle=0):
    overlay = cv2.resize(overlay,(0,0),fx=scale,fy=scale)
    h,w,_ = overlay.shape  # Size of foreground
    rows,cols,_ = src.shape  # Size of background Image
    y,x = pos[0],pos[1]    # Position of foreground/overlay image
    #Get rotation matrix
    r_rows,r_cols,_ = overlay.shape
    M = cv2.getRotationMatrix2D((r_cols/2,r_rows/2),angle,1)
    
    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            #rotation
            rx = int(M[0][0]*i + M[0][1]*j + M[0][2])
            ry = int(M[1][0]*i + M[1][1]*j + M[1][2])
            #print(i,",",j," have been changed to ",x+rx,",",y+ry)
            
            if x+rx >= rows or y+ry >= cols or x+rx<0 or y+ry<0: #Out of bounds
                continue
            alpha = float(overlay[i][j][3]/255.0)   #read the alpha channel 
            src[int(x+rx)][int(y+ry)] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+rx][y+ry]
    return src

#%% utility functions
def rect_to_xywh(a):
    return ( a.left(), a.top(), a.right(), a.bottom() )

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def get_eye_details(pred):
    left = pred[36]
    right = pred[45]
    #find vector
    vec = [right[0]-left[0], right[1]-left[1] ]
    pi = 3.14159
    angle = ( math.atan( vec[1] / vec[0] )*180) / pi
    #print("angle of face in degrees = ",angle)
    eye_dist = math.sqrt( pow(vec[0],2) + pow(vec[1],2) )
    return angle,eye_dist

#%% Main Function to apply dog filter
def add_dog_filter(img, dog_filter):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('Detector/shape_predictor_68_face_landmarks.dat')
    
    faces = detector(gray, 2)

    filter_nose = dog_filter['nose']
    filter_left_ear = dog_filter['ear_left']
    filter_right_ear = dog_filter['ear_right']
    
    if len(faces)==0:
        print("No faces Detected in the image")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for face in faces[::-1]: 
        (x, y, w, h) = rect_to_xywh(face)
        # add bounding box to color image
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        s = predictor(gray, face)
        s = shape_to_np(s)
        
        angle,eye_dist = get_eye_details(s);
        scale = 0.005*eye_dist
        
        nose = s[30]
        nose[0]-=(filter_nose.shape[1]/2)*scale*1
        nose[1]-=(filter_nose.shape[0]/2)*scale/1.5
        x = int(x - (filter_left_ear.shape[1]/2)*scale)
        w = int(w - (filter_left_ear.shape[1]/2)*scale)
        y = int(y - (filter_left_ear.shape[0])*scale)
        
        result = transparentOverlay(img.copy(),filter_nose,(nose[0],nose[1]),scale, angle)
        result = transparentOverlay(result.copy(),filter_left_ear,(x,y),scale,angle)
        result = transparentOverlay(result.copy(),filter_right_ear,(w,y),scale,angle)
        
        img = result      
    #Change to RGB
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    #and return
    return result

#%%  Main Program

def get_filter():
    #read filter image
    filter_path = 'filter3.png'
    filter_full = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED) #Read woth PNG transparency
        
    dog_filter = {  'nose' : filter_full[302:390,147:300],
                    'ear_left' : filter_full[55:195,0:160],
                    'ear_right' : filter_full[55:190,255:420],
                 }
    return dog_filter

#Testing
# img_path = 'Images/img5.jpg'
# result = add_dog_filter(img_path, dog_filter)

# plt.imshow(result)
# plt.show()
# result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
# cv2.imwrite('Images/Edited/'+img_path[7:] ,result)
