from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib.auth.models import User,auth
from . import models
from django.contrib import messages 
import io
import cv2
import librosa
import os
import numpy as np
import pickle
import base64
from django.http import JsonResponse
from .frsmfa.Authenticate import initiatemfa ,extract_audio,extract_frames
# Create your views here.
mfa=initiatemfa()

def loginPage(request):
    return render(request,'login.html')

def registerpage(request):
    return render(request,'signup.html')

def welcome(request):
    return render(request,'welcome.html')

def  Register(request):
    username = request.POST["username"]
    password = request.POST["password"]
    email=request.POST["email"]
    
    if User.objects.filter(username=username).exists():
        messages.info(request,'Username already taken')
        return redirect("/authenticate/RegisterPage")
    if User.objects.filter(email=email).exists():
        messages.info(request,'Email already exists')
        return redirect("/authenticate/RegisterPage")
    
    user = User.objects.create_user(username = username,password = password, email = email)
    user.save()
    #print("user created successfully")
    
    return render(request,"register.html",{'user':user})


def login(request):
    username = request.POST['username']
    password = request.POST['password']
    
    user = auth.authenticate(username=username,password = password)
    
    if user is not None:
        
        print("loggedin")
        print("yout otp  ",mfa.get_otp())
        return render(request,'check.html',{'user':user,})
    else:
        messages.info(request,'invalid credentials')
        return redirect('/authenticate/loginPage')
    
    
    
def voiceRegister(request):

    print('came')
    blob = request.FILES['video'].read();
    
    #converting binnary to intremediate format
    tempBytesFormat = io.BytesIO(blob)
    
    #getting back the  video from intermediate format
    with open('test.mpg',"wb") as outfile:
        outfile.write(tempBytesFormat.getbuffer())  
 
     #extracting audio
  
   
        
    file, extension = os.path.splitext('test.mpg')
    print('checking',file,extension)
        # Convert video into .wav file
    os.system('ffmpeg -i {file}{ext} {file}.wav'.format(file=file, ext=extension))    
    audio, samplerate = librosa.load('test.wav', sr = 16000) 
    os.remove('{}.wav'.format(file)) 
    
    if(len(audio)<32000):# less than 2 second
          return JsonResponse({'access':False},safe=False)
    
    userEmbedding  =  mfa.get_embedding_Array(audio)
    print(userEmbedding.shape)
    np_bytes = pickle.dumps(userEmbedding)
    np_base64 = base64.b64encode(np_bytes)

    
    models.VoiceData(userId = request.POST['userId'], userVoiceEmbedding = np_base64).save()
    
    
    
    return JsonResponse({'access':True},safe=False)
    


def authenticate(request):
    
    #ACCESSING form data
    print("hellowwwww")
    blob = request.FILES['video'].read();
    
    #converting binnary to intremediate format
    tempBytesFormat = io.BytesIO(blob)
    
    #getting back the  video from intermediate format
    with open('test.mp4',"wb") as outfile:
        outfile.write(tempBytesFormat.getbuffer())  
  
        
   
    
    #procesing video into images
    
   
    
    video = cv2.VideoCapture('test.mp4')
    sec = 0
    frameRate = 0.02
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
    print(len(frames))
    
    
     #extracting audio
  
    file, extension = os.path.splitext('test.mp4')
        # Convert video into .wav file
    os.system('ffmpeg -i {file}{ext} {file}.wav'.format(file=file, ext=extension))    
    audio, samplerate = librosa.load('test.wav', sr = 16000) 
    os.remove('{}.wav'.format(file)) 
    
    
    temp =  models.VoiceData.objects.get(userId = request.POST['userId'])
    np_bytes = base64.b64decode(temp.userVoiceEmbedding)

    enrolmentEmbedding =  pickle.loads(np_bytes)
    currentVideo = frames
    currentAudio = audio
    print("going")
    result = mfa.Authenticate(currentAudio,currentVideo,enrolmentEmbedding)
    result=True
    
    
    return  JsonResponse({'access':result},safe=False)
    