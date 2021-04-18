# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 19:58:07 2021

@author: mitran
"""

from torch.nn import Module,Linear,Dropout,LayerNorm,Conv1d,GroupNorm,ModuleList
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch import bmm as matmul
import torch
import numpy as np
import librosa
import math

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class GroupNormConvLayer(Module):
   def __init__(self,config,layer_id):
     super().__init__()
     self.conv= Conv1d(1, config.no_filters, kernel_size=(config.conv_kernel[layer_id],), stride=(config.conv_stride[layer_id],), bias=False)
     self.dropout= Dropout(p=0.0, inplace=False)
     self.layer_norm= GroupNorm( config.no_filters,  config.no_filters, eps=1e-05, affine=True)
     self.activation=gelu
   def forward(self,X):
     y=self.conv(X)
     y=self.dropout(y)
     y=self.layer_norm(y)    
     y=self.activation(y)
     return y
class ConvLayerNoNorm(Module):
  def __init__(self,config,layer_id):
     super().__init__()
     
     self.conv=Conv1d( config.no_filters, config.no_filters, kernel_size=(config.conv_kernel[layer_id],), stride=(config.conv_stride[layer_id],), bias=False)
     self.dropout=Dropout(p=0.0, inplace=False)
     self.activation=gelu

  def forward(self,X):
    y=self.conv(X)
    y=self.dropout(y)
    y=self.activation(y)

    return  y 

class PadLayer(Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states

class Convolutional_Embedding(Module):
    def __init__(self, config):
        super().__init__()
        self.conv = Conv1d(config.dimension,config.dimension,kernel_size=config.num_conv_pos_embeddings,padding=config.num_conv_pos_embeddings // 2,groups=config.num_conv_pos_embedding_groups,)
        self.conv = weight_norm(self.conv, name="weight", dim=2)
        self.padding = PadLayer(config.num_conv_pos_embeddings)
        self.activation = gelu

    def forward(self, X):
        X = X.transpose(1, 2)

        y = self.conv(X)
        y = self.padding(y)
        y = self.activation(y)

        y = y.transpose(1, 2)

        return y
    
    
class FeatureExtractor(Module):
  def __init__(self,config):
      # 1st layer is group norm and next 6 layers do not have norm (ccording to wav2vec2)
      super().__init__()
      Fe=[]
      Gnlayer=GroupNormConvLayer(config, layer_id=0)
      
      Fe.append(Gnlayer)
      for i in range(1,5):
          Fe.append(ConvLayerNoNorm(config, layer_id=2) )
      for i in range(5,7):
          Fe.append(ConvLayerNoNorm(config, layer_id=6))    
         

      self.conv_layers=ModuleList(Fe)
  def forward(self,X):
        y = X[:, None]
        for layer in self.conv_layers:
              y = layer(y)

        return y    
    


class FeatureProjection(Module): # convert the filters for each sample made by conv2d to projection(vector must represent the embedding of the sample)
    def __init__(self,config):

        super().__init__()
        self.layer_norm = LayerNorm(config.no_filters, eps=config.layer_norm_eps)
        self.projection = Linear(config.no_filters, config.dimension) # convert features for sample to embedding of the sample
        self.dropout =    Dropout(config.feat_extract_dropout)

    def forward(self, X):
        y = self.layer_norm(X)
        y = self.projection(y)
        y = self.dropout(y)

        return y


class Attention(Module):

    def __init__(
        self,
        dimension,num_heads,dropout
    ):
        super().__init__()
        self.num_heads = num_heads
        self.vec_dim = dimension // num_heads
        self.dropout=dropout
        self.scaling = self.vec_dim ** 0.5

        self.k_proj = Linear(dimension, dimension, bias=True)
        self.v_proj = Linear(dimension, dimension, bias=True)
        self.q_proj = Linear(dimension, dimension, bias=True)

        self.out_proj = Linear(dimension, dimension, bias=True)


    def split_heads(self,tensor):

         bz,seqlen,dim=tensor.shape
         assert(self.vec_dim*self.num_heads==dim)
         tensor=tensor.view(bz,seqlen,self.num_heads,self.vec_dim)
         tensor=tensor.transpose(1,2)
         tensor=tensor.view(bz*self.num_heads,seqlen,self.vec_dim)

         return tensor
    def concat_heads(self,tensor):

        bz_x_nohead,seqlen,vec_dim=tensor.shape
        assert(vec_dim==self.vec_dim)
        assert(bz_x_nohead%self.num_heads==0)
        bz= bz_x_nohead//self.num_heads    
        tensor=tensor.view(bz,self.num_heads,seqlen,vec_dim)
        tensor=tensor.transpose(1,2)
        tensor=tensor.reshape(bz,seqlen,self.num_heads*self.vec_dim)

        return tensor


    def forward(
        self,
        hidden_states
    ):

        
        bsz, seqlen, embed_dim = hidden_states.shape

        query_states = self.q_proj(hidden_states)
       
        key_states = self.k_proj(hidden_states)
        value_states =self.v_proj(hidden_states)


        query_states = self.split_heads(query_states)
        key_states =   self.split_heads(key_states)
        value_states = self.split_heads(value_states)

        attn_weights = matmul(query_states, key_states.transpose(1, 2)) / self.scaling            

        attn_probs = F.softmax(attn_weights, dim=-1)

        #attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = matmul(attn_probs, value_states)

        attn_output = self.concat_heads(attn_output)
           
        attn_output = self.out_proj(attn_output)

        return attn_output




class Feed_Forward(Module):
    def __init__(self,config):
      super().__init__()
      self.intermediate_dropout=Dropout(p=config.hidden_dropout_prob, inplace=False)
      self.intermediate_dense=Linear(config.dimension, config.intermediate_size, bias=True)
      self.activation=gelu
      self.output_dense=Linear(config.intermediate_size,config.dimension, bias=True)
      self.output_dropout=Dropout(p=config.hidden_dropout_prob, inplace=False)
    def forward(self,X):

      y=self.intermediate_dense(X)
      y=self.activation(y)
      y=self.intermediate_dropout(y)
      y=self.output_dense(y)
      y=self.output_dropout(y)
      
      return y        

class EncoderLayer(Module):
   def __init__(self,config):
      super().__init__() 
      self.attention=Attention(config.dimension,
          config.num_attention_heads,config.hidden_dropout_prob
          )
      self.dropout=Dropout(config.hidden_dropout_prob)
      self.feed_forward=Feed_Forward(config)
      self.layer_norm=LayerNorm(config.dimension, eps=config.layer_norm_eps)
      self.final_layer_norm=LayerNorm(config.dimension, eps=config.layer_norm_eps)

   def forward(self,X):
     
       y=self.attention(X)
       y=self.dropout(y)
       y=self.layer_norm(X+y)

       y1=self.feed_forward(y)
       y=self.final_layer_norm(y+y1)

       return y      

class Encoder(Module):
    def __init__(self,config):
        super().__init__()
        
        self.pos_conv_embed=Convolutional_Embedding(config)
        self.layer_norm = LayerNorm(config.dimension, eps=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.layers = ModuleList([EncoderLayer(config) for _ in range(config.no_encoders)])
    def forward(self,X):
        
        position_embeddings = self.pos_conv_embed(X)
        #print(X.shape,position_embeddings.shape)
        y = X + position_embeddings
        y = self.layer_norm(y)
        y = self.dropout(y)

        for layer in self.layers:            
            y = layer(y)
            
        return y 


class Transformer(Module):
  def __init__(self,config):
    super().__init__()
    self.feature_extractor=FeatureExtractor(config)
    self.feature_projection=FeatureProjection(config)
    self.encoder=Encoder(config)
  def forward(self,X):
    
    y=self.feature_extractor(X)
    # feature projector takes the filters(512) of the sample in time domain , the position(i) for 512 filters  are converted to the vector projection for the sample (i)
    y=y.transpose(1,2)# change the filter to latent dimension
    y=self.feature_projection(y)
    y=self.encoder(y)

    return y

class SpeechRecognition(Module):
    def __init__(self, config):
        super().__init__()

        self.wav2vec2 = Transformer(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.lm_head = Linear(config.dimension, config.vocab_size)


    def forward( self,X):
       

        y = self.wav2vec2(X)
        y = y[0]
        y = self.dropout(y)

        y = self.lm_head(y)
        return y   
    def predict(self,X):
        with  torch.no_grad():
              op=self.forward(torch.tensor(X,dtype=torch.float))
              op=op.detach()
        predicted_ids = torch.argmax(op, dim =-1)
        
        return predicted_ids
    
def wav2vecmodel(config):
   model= SpeechRecognition(config)
   model.to(torch.device('cpu'))
   model.eval()
   return model

class  srdecoder():
  def __init__(self):


      self.vocab_dict={0 :'' ,  
            1  : '<s>',
            2  : '</s>',
            3  : '<unk>',
            4  : ' ',
            5  : 'E',
            6  : 'T',
            7  : 'A',
            8  : 'O',
            9  : 'N',
            10 : 'I',
            11  : 'H',
            12  : 'S',
            13  : 'R',
            14  : 'D',
            15  : 'L',
            16  : 'U',
            17  : 'M',
            18  : 'W',
            19  : 'C',
            20  : 'F',
            21  : 'G',
            22  : 'Y',
            23  : 'P',
            24  : 'B',
            25  : 'V',
            26  : 'K',
            27  : "'",
            28  : 'X',
            29  : 'J',
            30  : 'Q',
            31  : 'Z'}

  def remove_adjacents(self,lst):
     new_list=[]
     for i in range(len(lst)-1):
     
        if(lst[i]!=lst[i+1]):
            new_list.append(lst[i].item())
     return   new_list  
  def token2word(self,tokens):
     sentence=[]
     for t in tokens:
       sentence.append(self.vocab_dict[t])
     return ''.join(sentence) 
  def decode(self,predictedid):
      
      tok=self.remove_adjacents(predictedid)
      words=self.token2word(tok).lower()
      return words
     

