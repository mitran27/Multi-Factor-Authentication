# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:57:07 2021

@author: mitran
"""
from torch.nn import Module,Sequential,init
import torch.nn.functional as F

import torch
from torch.nn import GRU,MaxPool1d,Dropout,LeakyReLU,Linear,BatchNorm1d,Flatten,Conv1d,Softmax,Conv2d,LSTM,AdaptiveAvgPool2d,Parameter
import numpy as np
from torch import nn

import math






class ArcMarginProduct(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))

   
class Angular(Module):

    def __init__(self, embedding_size, no_Classes):
      
        super(Angular, self).__init__()
        
        self.arc_margin_product  =ArcMarginProduct(embedding_size,no_Classes)
        self.head=Linear(embedding_size, no_Classes, bias=False)
     
  



       


  

class Convolution_block(Module):
  def __init__(self,filters):
      super().__init__()
      self.conv1=Conv1d(1,filters,kernel_size=(3),padding=0,stride=3)
      self.norm1=BatchNorm1d(num_features=filters)
      self.act1=LeakyReLU()

  def forward(self,X):

     y=self.conv1(X)
     y=self.norm1(y)
     y=self.act1(y)

     return y   

class Identiy(Module):
   def __init__(self):
      super().__init__()
      pass
   def forward(self,X):
        return X   
    
    
class Resnet1D(Module):
   def __init__(self,filters):
      super().__init__()
      self.conv1=Conv1d(filters[0],filters[1],kernel_size=(3),padding=1)
      self.norm1=BatchNorm1d(num_features=filters[1])
      self.act1=LeakyReLU()

      self.conv2=Conv1d(filters[1],filters[1],kernel_size=(3),padding=1)
      self.norm2=BatchNorm1d(num_features=filters[1])

      if(filters[0]!=filters[1]): 
           self.convr=Conv1d(filters[0],filters[1],kernel_size=(1),padding=0)
      else:
          self.convr=Identiy()     
      self.normr=BatchNorm1d(num_features=filters[1])
      self.actr=LeakyReLU()

      self.pool=MaxPool1d(3)



   def forward(self,X):

     y=self.conv1(X)
     y=self.norm1(y)
     y=self.act1(y)

     y=self.conv2(y)
     y=self.norm2(y)

     res=self.convr(X)
     res=self.normr(res)

     y=torch.add(y,res)
     y=self.actr(y)

     y=self.pool(y)  
     

     return y   

class Utterance_Block(Module):
   def __init__(self,in_dim,out_dim):
     
      super().__init__()
      self.bn_before_gru = BatchNorm1d(in_dim)
      self.relu=LeakyReLU(negative_slope = 0.3)
      self.l1=Linear(in_dim,512)
      self.rnn=GRU(512,512,bidirectional=True)

      self.l2=Linear(1024,out_dim)
      self.norm1=BatchNorm1d(num_features=out_dim)

   def forward(self,X):


    y=self.bn_before_gru(X)
    y=self.relu(y)
   
    y = y.permute(0,2, 1)
    b=y.shape[2]
    y=self.l1(y)
    self.rnn.flatten_parameters()    
    y,_=self.rnn(y)    
    y=torch.sum(y,1)
    y=torch.div(y,b)
    y=self.l2(y) 
    

   

    return y
class Rawnet(Module):
   def __init__(self,config):
     super().__init__() 
     self.block1=Convolution_block(128)

     self.block2=Resnet1D([128,128])
     self.block3=Resnet1D([128,128]) 

     self.block4=Resnet1D([128,256])
     self.block5=Resnet1D([256,256])
     self.block6=Resnet1D([256,512])
     self.block7=Resnet1D([512,512])

     self.utter_blk=Utterance_Block(512,512)
     self.classs=config.no_clas
     self.embedding_dimension=config.embedding_dimension
     self.lossfn=Angular(config.embedding_dimension,config. no_clas)
     
     
   
    
   def accuracy(self,predictions, labels):
      classes = torch.argmax(predictions, dim=1)
      return torch.mean((classes == labels).float())   
  
   def extract_Embedding(self,X):
       #print('audshape',X.shape)
       with  torch.no_grad():
          embedding=self.forward(torch.tensor(X,dtype=torch.float))
       return embedding
       
   def forward(self,X):

     y=self.block1(X)

     y=self.block2(y)  
     y=self.block3(y) 
 
     y=self.block4(y)  
     y=self.block5(y)  
     y=self.block6(y)  
     y=self.block7(y)  

     features=self.utter_blk(y)  

     
     return features 














class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)
        

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

      
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

 


    def forward(self, waveforms):
        

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) 
class Identiy(Module):
   def __init__(self):
      super().__init__()
      pass
   def forward(self,X):
        return X   

class Residual_block(nn.Module):
    def __init__(self, nb_filts, first = False):
        super(Residual_block, self).__init__()


        self.first = first        
        if not self.first : self.bn1 = nn.BatchNorm1d(num_features = nb_filts[0])  

        self.lrelu = nn.LeakyReLU()
        self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv1d(in_channels = nb_filts[0],out_channels = nb_filts[1],kernel_size = 3,padding = 1,stride = 1)
        self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels = nb_filts[1],out_channels = nb_filts[1],kernel_size = 3,padding = 1,stride = 1)
        
        if nb_filts[0] != nb_filts[1]:
            self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],out_channels = nb_filts[1],padding = 0,kernel_size = 1,stride = 1)
            
        else:
            self.conv_downsample = Identiy()

        self.mp = nn.MaxPool1d(3)
        
    def forward(self, x):
        identity = x
       
            
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu_keras(out)
        out = self.conv2(out)
        
        identity = self.conv_downsample(identity)
            
        out += identity
        out = self.mp(out)
        return out


class RawNet2(nn.Module):
    def __init__(self):
        super(RawNet2, self).__init__()

        self.ln = LayerNorm(59049)
        self.first_conv = SincConv_fast(in_channels = 1,out_channels = 128,kernel_size =251)

        self.first_bn = nn.BatchNorm1d(num_features = 128)
        self.lrelu = nn.LeakyReLU()
        self.lrelu_keras = nn.LeakyReLU(negative_slope = 0.3)
        
        self.block0 = nn.Sequential(Residual_block(nb_filts = [128,128], first = True))
        self.block1 = nn.Sequential(Residual_block(nb_filts = [128,128]))
 
        self.block2 = nn.Sequential(Residual_block(nb_filts =[128,256]))
        self.block3 = nn.Sequential(Residual_block(nb_filts = [256,256]))
        self.block4 = nn.Sequential(Residual_block(nb_filts = [256,256]))
        self.block5 = nn.Sequential(Residual_block(nb_filts = [256,256]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc_attention0 = nn.Sequential(nn.Linear(in_features = 128,out_features = 128))
        self.fc_attention1 = nn.Sequential(nn.Linear(in_features = 128,out_features = 128))
        self.fc_attention2 = nn.Sequential(nn.Linear(in_features = 256,out_features = 256))
        self.fc_attention3 = nn.Sequential(nn.Linear(in_features = 256,out_features = 256))
        self.fc_attention4 = nn.Sequential(nn.Linear(in_features = 256,out_features = 256))
        self.fc_attention5 = nn.Sequential(nn.Linear(in_features = 256,out_features = 256))

        self.bn_before_gru = nn.BatchNorm1d(num_features = 256)
        self.gru = nn.GRU(input_size = 256,hidden_size = 1024,num_layers = 1,batch_first = True)

        
        self.fc1_gru = nn.Linear(in_features = 1024,out_features = 1024)
        self.fc2_gru = nn.Linear(in_features = 1024,out_features =6112,bias = True)

    def extract_Embedding(self,X):
       #print('audshape',X.shape)
       with  torch.no_grad():
          embedding=self.forward(torch.tensor(X,dtype=torch.float))
          embedding=embedding.detach()
          #print(embedding.shape)
       return embedding


    def forward(self, x, y = 0, is_test=True):


      
        x = self.ln(x)
        x = F.max_pool1d(torch.abs(self.first_conv(x)), 3)
        x = self.first_bn(x)
        x = self.lrelu_keras(x)
        
        x0 = self.block0(x)
        
        y1 = self.avgpool(x0)
        y1 = torch.squeeze(y1,-1) 
        y1 = self.fc_attention0(y1)
        y1 = torch.sigmoid(y1)
        y1 = torch.unsqueeze(y1,-1)

        x = x0 * y1 + y1    

        x1 = self.block1(x)
        y1 = self.avgpool(x1)
        y1 = torch.squeeze(y1,-1) 
        y1 = self.fc_attention1(y1)
        y1 = torch.sigmoid(y1)
        y1 = torch.unsqueeze(y1,-1)

        x = x1 * y1 + y1
        
        x2 = self.block2(x)
        y1 = self.avgpool(x2)
        y1 = torch.squeeze(y1,-1) 
        y1 = self.fc_attention2(y1)
        y1 = torch.sigmoid(y1)
        y1 = torch.unsqueeze(y1,-1)

        x = x2 * y1 + y1 

        x3 = self.block3(x)
        y1 = self.avgpool(x3)
        y1 = torch.squeeze(y1,-1) 
        y1 = self.fc_attention3(y1)
        y1 = torch.sigmoid(y1)
        y1 = torch.unsqueeze(y1,-1)

        x = x3 * y1 + y1 
        x4 = self.block4(x)
        y1 = self.avgpool(x4)
        y1 = torch.squeeze(y1,-1) 
        y1 = self.fc_attention4(y1)
        y1 = torch.sigmoid(y1)        
        y1 = torch.unsqueeze(y1,-1)

        x = x4 * y1 + y1 

        x5 = self.block5(x)

        y1 = self.avgpool(x5)
        y1 = torch.squeeze(y1,-1) 
        y1 = self.fc_attention5(y1)
        y1 = torch.sigmoid(y1)
        y1 = torch.unsqueeze(y1,-1)

        x = x5 * y1 + y1 

        x = self.bn_before_gru(x)
        x = self.lrelu_keras(x)

        x = x.permute(0, 2, 1)

        self.gru.flatten_parameters()

        x, _ = self.gru(x)
        x = x[:,-1,:]

        code = self.fc1_gru(x)

        if is_test: return code
        
        
        else:
          
           code_norm = code.norm(p=2,dim=1, keepdim=True) / 10.
           code = torch.div(code, code_norm)
           out = self.fc2_gru(code) 
           return out
    


  


  
    














    
def spkrnetmodel(config):
   model= RawNet2()
   model.to(torch.device('cpu'))
   model.eval()
   return model 


class spkrdecoder():
    def __init__(self):
        
        
        self.tot=59049
    def padding(self,audio):
        if(len(audio)>self.tot):
            audio=audio[4000:self.tot]
        if(len(audio<self.tot)):
            padlen=self.tot-len(audio)
            pad=audio[:padlen]
            audio=np.hstack([pad,audio])
        return audio    
            

            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    