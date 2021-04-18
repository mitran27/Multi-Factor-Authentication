# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:57:40 2021

@author: mitran
"""

import tensorflow as tf
from tensorflow.keras.layers import *
from keras import backend as F
from tensorflow.keras.utils import Sequence 
from tensorflow.keras.models import Model as Module
from tensorflow.keras.optimizers import Adam
import keras
import zipfile
from tqdm.notebook import tqdm
import numpy as np
import os


class CTCLayer(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn =F.ctc_batch_cost

    def call(self, y_true, y_pred,input_length,label_length):     
        
        y_pred = y_pred[:, 2:, :]
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        return loss
    
    
def Lipnet(config):
    input_data = Input(name='the_input', shape=config.input_shape, dtype='float32')
    labels = Input(name='the_labels', shape=[config.seqlen], dtype='float32')       
    input_length = Input(name='input_length', shape=[1], dtype='int64')   
    label_length = Input(name='label_length', shape=[1], dtype='int64')    
     
   
   
   
    stcnn1_padding = ZeroPadding3D(padding=(1,2,2))(input_data) 
    stcnn1_convolution = Conv3D(16, (3, 5, 5), strides=(1,2,2), kernel_initializer='he_uniform')(stcnn1_padding)
    stcnn1_bn = BatchNormalization()(stcnn1_convolution)
    stcnn1_acti = Activation('relu')(stcnn1_bn)
    stcnn1_dp = SpatialDropout3D(0.2)(stcnn1_acti)
    stcnn1_maxpool = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),name='stcn1')(stcnn1_dp)

    stcnn2_padding = ZeroPadding3D(padding=(1,2,2))(stcnn1_maxpool)
    stcnn2_convolution = Conv3D(32, (3, 5, 5), strides=(1,2,2), kernel_initializer='he_uniform')(stcnn2_padding)
    stcnn2_bn = BatchNormalization()(stcnn2_convolution)
    stcnn2_acti = Activation('relu')(stcnn2_bn)
    stcnn2_dp = SpatialDropout3D(0.2)(stcnn2_acti)
    stcnn2_maxpool = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),name='stcn2')(stcnn2_dp)

    stcnn3_padding = ZeroPadding3D(padding=(1,2,2))(stcnn2_maxpool)
    stcnn3_convolution = Conv3D(64, (3, 3, 3), strides=(1,2,2), kernel_initializer='he_uniform')(stcnn3_padding)
    stcnn3_bn = BatchNormalization()(stcnn3_convolution)
    stcnn3_acti = Activation('relu')(stcnn3_bn)
    stcnn3_dp = SpatialDropout3D(0.2)(stcnn3_acti)
    stcnn3_maxpool = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),name='stcn3')(stcnn3_dp)

    stcnn3_maxpool_flatten = TimeDistributed(Flatten(),name='features')(stcnn3_maxpool)

    gru_1 = GRU(config.dimension, return_sequences=True, name='gru1_a')(stcnn3_maxpool_flatten)
    gru_1b = GRU(config.dimension, return_sequences=True, go_backwards=True, name='gru1_b')(stcnn3_maxpool_flatten)
    gru1_merged = concatenate([gru_1, gru_1b], axis=2,name='gru1')
    gru1_dropped = gru1_merged

    gru_2 = GRU(config.dimension, return_sequences=True, name='gru2_a')(gru1_dropped)
    gru_2b = GRU(config.dimension, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_dropped)
    gru2_merged = concatenate([gru_2, gru_2b], axis=2,name='gru2')
    gru2_dropped = gru2_merged
    #fc linear layer



    y_pred=Dense(config.no_class, activation="softmax", name="dense2")(gru2_dropped)

    loss_out  = CTCLayer(name="ctc")(labels, y_pred,input_length,label_length)

    model = Module(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])


    return model

def lipnetmodel(config):
    
    lip_model=Lipnet(config)
    prediction_model =Module(
    lip_model.get_layer(name="the_input").input, lip_model.get_layer(name="dense2").output
)
    return prediction_model


import dlib
import cv2



class vsrdecoder():
   def __init__(self,weightspath):

     self.predictor = dlib.shape_predictor(weightspath+'face68.dat')
     self.detector = dlib.get_frontal_face_detector()
     self.aligner = dlib.shape_predictor(weightspath+'face5.dat')
     
     
     import string
     x=string.ascii_lowercase[:26]
     decodestr={}
     for i in x:
      decodestr[ord(i)-ord('a')]=i
     decodestr[26]=' ' 
     self.decodestr=decodestr
      
   def decode_batch_predictions(self,pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results =F.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :100
    ]
    output_text = []
    for res in results:
        string=np.array((res))
        resstr =''.join([self.decodestr[i] for i in string if i>=0])
        output_text.append(resstr)
    return output_text

   def align(self,image,detect_face): 
 
       img_shape = self.aligner(image, detect_face) 
       aligned = dlib.get_face_chip(image, img_shape)

       return aligned

   def extract_lips(self,video,shape=(100,50)):
    
    frames=[]
    for vidimage in video:
        
        if(len(self.detector(vidimage,1))==0):
                 
                 continue

        
        detect_face=self.detector(vidimage,1)[0]# select first face
        
        image=self.align(vidimage,detect_face)
          
        if(len(self.detector(image,1))==0):
              
              detect_face= dlib.rectangle(left=0, top=0, right=150, bottom=150)
        else:              
               detect_face=self.detector(image,1)[0]
          
        points=np.array([[p.x, p.y] for i,p in enumerate(self.predictor(image, detect_face).parts())])[48:] # get mouth region
          
          
        left_m=min(points[:,0])-10
        right_m=max(points[:,0])+10 # select mouth 10 pixels extra
          
        mouth_center=np.mean(points[:, -2:], axis=0) #get mouth center
    
        scale=shape[0]/(right_m-left_m) # get scale for our image shape from video's shape
          
        mouth_center=mouth_center*scale
          
        left_m=int(mouth_center[0]-shape[0]//2)
        right_m=int(mouth_center[0]+shape[0]//2)
        up_m=int(mouth_center[1]-shape[1]//2)
        down_m=int(mouth_center[1]+shape[1]//2)
    
        scaled_image=cv2.resize(image,(int(image.shape[1] * scale), int(image.shape[0] * scale)))
          
        mouth_Crop= scaled_image[up_m:down_m, left_m:right_m,:]  # crop the mouth
        if(mouth_Crop.shape!=(50, 100, 3)):
              mouth_Crop=cv2.resize(mouth_Crop,(100, 50))
         
        frames.append(mouth_Crop)
    return frames   
    