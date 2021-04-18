# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 20:02:26 2021

@author: mitran
"""


class w2vconfig():
 
   def __init__(self):
        self.vocab_size=32
        self.dimension=768
        self.no_encoders=12
        self.num_attention_heads=12
        self.intermediate_size=3072
        self.hidden_dropout_prob=0.0
        self.attention_probs_dropout_prob=0.0, 
        self.layer_norm_eps=1e-5
        self.feat_extract_dropout=0.0
        self.no_filters=512
        self.conv_stride=(5, 2, 2, 2, 2, 2, 2)
        self.conv_kernel=(10, 3, 3, 3, 3, 2, 2)
        self.conv_bias=False
        self.num_conv_pos_embeddings=128
        self.num_conv_pos_embedding_groups=16   
        
        
        
class lipconfig:
  def __init__(self):
    self.no_class=28
    self.input_shape=(None,50,100,3)
    self.seqlen=(None)
    self.dimension=64        
    
class spkrconfig:
  def __init__(self):
    self.embedding_dimension=512
    self.no_clas=1000    