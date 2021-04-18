# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 07:48:29 2021

@author: mitran
"""


import os

from models.speechrecognition import wav2vecmodel,srdecoder
from models.speakerrecognition import spkrnetmodel,spkrdecoder
from models.lipreading import lipnetmodel,vsrdecoder

from modelconfigs import lipconfig,w2vconfig,spkrconfig





import tensorflow as tf
import torch
import torch.nn.functional as F


import io
import cv2
import librosa


import numpy as np
import pickle
import base64
import random
import numpy as np

#/Authenticate/frsmfa


class mfaconfig():
    def __init__(self):
        self.models={
            
            'speech' : wav2vecmodel(w2vconfig()),
            'speaker':spkrnetmodel(spkrconfig()),
            'lip':lipnetmodel(lipconfig())
            }
        self.weights={
            'lip':'./weights/lipnet.h5',
            'speech' :  './weights/w2v.pth',
            'speaker':'./weights/rawnet2.pt'
            }
        self.decoders={
            'speech':srdecoder(),
            'lip' :  vsrdecoder('./weights/'),
            'speaker':spkrdecoder()
            }
            
            
def levenshtein(s, t):
          
            rows = len(s)+1
            cols = len(t)+1
            distance = np.zeros((rows,cols),dtype = int)
         
            for i in range(1, rows):
                distance[i][0] = i
            for k in range(1,cols):            
                distance[0][k] = k
           
            for col in range(1, cols):
                for row in range(1, rows):
                    if s[row-1] == t[col-1]:
                        cost = 0 
                    else:
                        cost = 2
                      
                    distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                         distance[row][col-1] + 1,          # Cost of insertions
                                         distance[row-1][col-1] + cost)     # Cost of substitutions
            Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
            return Ratio       
    
class MultifactorAuthentication():
      def __init__(self,config):
        self.sr=config.models['speech']
        self.vsr=config.models['lip']
        self.spkr=config.models['speaker']
    
        
        
        
        
        self.sr.load_state_dict(torch.load(config.weights['speech']))
        self.spkr.load_state_dict(torch.load(config.weights['speaker'], map_location=torch.device('cpu')))
        self.vsr.load_weights(config.weights['lip'])
        
        
    
        self.speechdecoder=config.decoders['speech']
        self.lipdecoder=config.decoders['lip']
        self.spkrdecoder=config.decoders['speaker']
        
        self.sr.eval()
        
        self.otp='bin blue at f two now'
        self.otplist=['bin blue at f two now']
        print("FRS labs Multi factor Authentication initiated")

      def get_otp(self):
          self.otp= random.choice(self.otplist)
          return self.otp
      def get_embedding(self,audio):
          
          
         if(len(audio.shape)==1):
             audio=audio.reshape(1,1,audio.shape[0])
         embedding=self.spkr.extract_Embedding(audio)
         return embedding
      def get_embedding_Array(self,audio):  
         """ 
         import soundfile as sf
         sf.write('file4s.wav', audio, 16000 ) """
          
         no_segments=audio.shape[0]//59049  #each segment 4 seconds
         segments=np.split(audio[:no_segments*59049],no_segments)
         embeddingarray=[]
         print("total segments",len(segments))
         for n,aud in enumerate(segments):
             print(n)
             emb=self.get_embedding(aud).numpy()
             embeddingarray.append(emb)
         arr=np.array((embeddingarray))  
         return arr  

         
      def Authenticate(self,testaudio,testvideo,enrollmentembedding):
          
          speechresult=self.speechauthenticate(testaudio)
          
          lipresult=self.lipauthenticate(testvideo)
          
          spkrresult=self.speakerauthenticate(testaudio,enrollmentembedding)
          print(speechresult,lipresult,self.otp)
          print(spkrresult)
          
          return [levenshtein(speechresult, self.otp),levenshtein(lipresult, self.otp),spkrresult]
          
          
      def  speechauthenticate(self,testaudio):
          aud=np.array((testaudio)).reshape(1,len(testaudio))
          text=self.sr.predict(aud)
          result=self.speechdecoder.decode(text)
          
          return result
      def speakerauthenticate(self,testaudio,enrollmentembedding):
          
          
          paddedaudio=self.spkrdecoder.padding(testaudio)
          #import soundfile as sf
          #sf.write('file4s.wav', paddedaudio, 16000 )
          
          testembedding=self.get_embedding(paddedaudio)
          emb2=testembedding
          results=[]
          
          for emb1 in enrollmentembedding:    
              #print(F.cosine_similarity(emb1, emb2,-1, eps=1e-6))
              #print(type(emb1),emb1.shape,type(emb2),emb2.shape)
              score=F.cosine_similarity(torch.tensor(emb1,dtype=torch.float), emb2,-1, eps=1e-6)
              results.append(score)
              
              
          return results    

          
          
          
          
      def lipauthenticate(self,testvideo):
        print(len(testvideo))  
        lip_frames= self.lipdecoder.extract_lips(testvideo)  
        #print(len(lip_frames))
        
          
       
            
        lip_frames=np.array((lip_frames))/255.0
        print('lipshape',lip_frames.shape)
        if(len(lip_frames)<40 or len(lip_frames.shape)<4):# less than 40 seconds or not in video format
          return ' '
        preds=self.vsr.predict(tf.expand_dims(lip_frames,0)) 
        word=self.lipdecoder.decode_batch_predictions(preds)[0]
        return word
    
      
          
def initiatemfa():
    mfa=MultifactorAuthentication(mfaconfig())
    return mfa



def extract_frames(path,fps=0.02):
    video = cv2.VideoCapture(path)
    sec = 0
    frameRate = fps
    count=1
    frame_available = True
    frames=[]
    
    while frame_available:
        sec+=frameRate
        sec=round(sec,2)
        video.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        frame_available,vidimage = video.read()
    
        if(frame_available):
            frames.append(vidimage)
    video.release()
    return frames
os.system('ffmpeg -hide_banner -loglevel error')      
def extract_audio(filename):
    
    
    file, extension = os.path.splitext(filename)
    #nfile=file.split('/')[-1]
    x=os.system('ffmpeg -i {file}{ext} {file}.wav'.format(file=file, ext=extension))
    audio, samplerate = librosa.load(file+'.wav', sr = 16000) 
    os.remove('{}.wav'.format(file))  
    
    return audio
 



frames=extract_frames('test2.mp4')
          
mfa=MultifactorAuthentication(mfaconfig())
      
         
         

  
filename='enroll.mp4'

audio=extract_audio(filename)

enrollementembedding=mfa.get_embedding_Array(audio)


filename='test2.mp4'  



audio=extract_audio(filename)
frames=extract_frames(filename)

currentVideo = frames
currentAudio = audio


print(currentAudio.shape)

from datetime import datetime

t1=datetime.now()

result=mfa.Authenticate(currentAudio,currentVideo,enrollementembedding)

t2=datetime.now()
print('time taken   ',t2-t1)


print("Speech:",result[0],"lip ",result[1],"speaker  ",max(result[2]))
        
"""          
          



path='./check/'

files=os.listdir(path)



import random

random.shuffle(files)

for filename in files:
    file1 = open("myfile.txt","a")#append mode

    if(filename.endswith('mpg')==False):
        continue
    
    
    audio=extract_audio(path+filename)
    frames=extract_frames(path+filename)
    
    
    speechtext=mfa.speechauthenticate(audio)
    liptext=mfa.lipauthenticate(frames)
    txtname=filename.replace('mpg','txt')
    
    groundtruth=open(path+txtname).read()
    
    file1.write("\n\n\nresult for file  "+filename)
    
    file1.write("\n\nSpeech result :")
    
    file1.write('\ngroundtruth  :'+groundtruth)
    file1.write('\nprediction  :'+speechtext)
    file1.write('\nscore  '+str(levenshtein(groundtruth,speechtext)))
    
    file1.write("\n\nlip result :")
    
    file1.write('\ngroundtruth  :'+groundtruth)
    file1.write('\nprediction  :'+liptext)
    file1.write('\nscore  '+str(levenshtein(groundtruth,liptext)))
    file1.close()
   


  
audio=extract_audio(filename)
frames=extract_frames(filename)

currentVideo = frames
currentAudio = audio


print(currentAudio.shape)



#mfa.Authenticate(currentAudio,currentVideo,enrollementembedding)

"""
"""


x=lipnetmodel(lipconfig())
x.load_weights('./weights/lipnet.h5')

y=wav2vecmodel(w2vconfig()).to('cpu')
y.load_state_dict(torch.load('./weights/w2v.pth'))

z=spkrnetmodel(spkrconfig()).to('cpu')
z.load_state_dict(torch.load('./weights/spkrfinal.pth'))"""


