import torch
from torch import nn
import numpy as np

class EnhanceModel(nn.Module):
    def __init__(self,discriminator,generator):
        super(EnhanceModel,self).__init__()
        
        self.discriminator = discriminator
        self.generator = generator
        
        
        
    def forward(self,f):
        x