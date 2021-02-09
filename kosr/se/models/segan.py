import torch
from torch import nn
import numpy as np
from layers.vbnorm import VirtualBatchNorm1d

class Discriminator(nn.Module):
    def __init__(self,conf):
        super(Discriminator,self).__init__()
        self.dis_n_layers = conf['dis_n_layers']
        #self.in_c = conf['in_channel']
        #self.out_c = conf['out_channel']
        self.in_c = 2
        self.out_c = 32
        
        self.k_s = 31
        self.stride = 2
        self.padding = 15
        negative_slope = 0.03
        
        self.conv_first = nn.Conv1d(self.in_c, self.out_c, self.k_s, self.stride, self.padding)
        self.vbn_first = VirtualBatchNorm1d(self.out_c)
        self.lrelu_first = nn.LeakyReLU(negative_slope)
        self.conv_layers = []
        self.vbnorms = []
        self.lrelus = []
        for i in range(1,self.dis_n_layers-1):
            if i%2==0:
                self.conv_layers.append(nn.Conv1d(self.out_c, self.out_c, self.k_s, self.stride, self.padding))
                self.vbnorms.append(VirtualBatchNorm1d(self.out_c))
                self.lrelus.append(nn.LeakyReLU(negative_slope))
            else:
                self.conv_layers.append(nn.Conv1d(self.out_c, self.out_c*2, self.k_s, self.stride, self.padding))
                self.out_c *= 2
                self.vbnorms.append(VirtualBatchNorm1d(self.out_c))
                self.lrelus.append(nn.LeakyReLU(negative_slope))
        self.conv_final = nn.Conv1d(self.out_c,1,kernel_size=1,stride=1)
        self.lrelu_final = nn.LeakyReLU(negative_slope)
        
        self.fully_connected = nn.Linear(in_features=8,out_features=1) #need to be modifiesd
        #self.sigmoid = nn.Sigmoid()
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
                
    def forward(self,data):
        means = []
        
        #reference pass
        x , ref_x = data
        ref_x = self.conv_first(ref_x)
        ref_x, mean1, meansq1 = self.vbn_first(ref_x, None, None)
        ref_x = self.lrelu_first(ref_x)
        for i in range(0,self.dis_n_layers-2):
            ref_x = self.conv_layers[i](ref_x)
            ref_x, mean, meansq = self.vbnorms[i](ref_x, None, None)
            ref_x = self.lrelus[i](ref_x)
            means.append([mean,meansq])
        
        #train pass
        x = self.conv_first(x)
        x, _, _ = self.vbn_first(x, means[0,0], means[0,1])
        x = self.lrelu_first(x)
        for i in range(0,self.dis_n_layers-2):
            x = self.conv_layers[i](x)
            x, _, _ = self.vbnorms[i](x, means[i,0], means[i,1])
            x = self.lrelus[i](x)
        x = self.conv_final(x)
        feat = self.lrelu_final(x)
        # reduce down to a scalar value
        x = torch.squeeze(feat)
        x = self.fully_connected(x)
        # return self.sigmoid(x)
        return x, feat 
        
    
        
class Generator(nn.Module):
    def __init__(self,conf):
        super(Generator,self).__init__()
        self.gen_n_layers = conf['gen_n_layers']
        self.in_c = 1
        self.out_c = 16
        self.k_s = 32
        self.stride = 2
        self.padding = 15
        
        self.enc_first = nn.Conv1d(self.in_c, self.out_c, self.k_s, self.stride, self.padding)
        self.enc_nl_first = nn.PReLU()
        self.enc_convs = []
        self.enc_nls = []

        #enc init
        for i in range(1,self.gen_n_layers-1):
            if i%2==0:
                self.enc_convs.append(nn.Conv1d(self.out_c, self.out_c, self.k_s, self.stride, self.padding))
                self.enc_nls.append(nn.PReLU())
            else:
                self.enc_convs.append(nn.Conv1d(self.out_c, self.out_c*2, self.k_s, self.stride, self.padding))
                self.out_c *= 2
                self.enc_nls.append(nn.PReLU())
        
        self.dec_first = nn.ConvTranspose1d(self.out_c*2, self.out_c//2, self.k_s, self.stride, self.padding)
        self.dec_nl_first = nn.PReLU()
        self.dec_convs = []
        self.dec_nls = []
        self.out_c //= 2
        
        #dec init
        for i in range(1,self.gen_n_layers-1):
            if i%2==0:
                self.dec_convs.append(nn.ConvTranspose1d(self.out_c*2, self.out_c//2, self.k_s, self.stride, self.padding))
                self.dec_nls.append(nn.PReLU())
            else:
                self.dec_convs.append(nn.ConvTranspose1d(self.out_c, self.out_c//2, self.k_s, self.stride, self.padding))
                self.out_c //= 2
                self.dec_nls.append(nn.PReLU())
                
        self.dec_final = nn.ConvTranspose1d(32, 1, 32, 2, 15)
        self.dec_tanh = nn.Tanh()
        
        self.init_weights()
        
    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)
                
    def forward(self,x):
        
        e = []
        e.append(self.enc_first(x))
        x = self.enc_nl_first(e[0])
        
        #encoding step
        for i in range(0,self.gen_n_layers-2):
            e.append(self.enc_convs[i](x))
            x = self.enc_nls[i](e[i+1])
        # x = compressed feature, the 'thought vector'
                     
        # concatenate the thought vector with latent variable
        encoded = torch.cat((x, torch.randn_like(x).to(x)), dim=1)
        
        #decoding step
        d = self.dec_first(encoded)
        
        # d_c : concatenated with skip-connected layer's output & passed nonlinear layer
        d_c = self.dec_nl_first(torch.cat((d,e[-2]),dim=1))
        for i in range(0,self.gen_n_layers-2):
            d = self.dec_convs[i](d_c)
            d_c = self.dec_nls[i](torch.cat((d,e[-1*(i+1)-2]),dim=1))
            
        out = self.dec_tanh(self.dec_final(d_c))
        
        return out
                               
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        