
# -*- coding: utf-8 -*-
"""
@author: Haiyang.Tang
Do not used for commercial purpose!
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#Transformer Positional Encoder

""" 
    PositionalEncoding: positional encoding method in original natural language processing.
    This methdo will not be used! But leave it for further research.
    
    
    Methods
    -------
    forward(x):
        add positional encoding to x and return x.
"""


class PositionalEncoding(nn.Module):
    # custom code
    def __init__(self,num_features, sequence_len=6, d_model=9):
        super(PositionalEncoding, self).__init__()
        if torch.cuda.is_available():
            self.device='cuda:0'
        else:
            self.device='cpu'
        pe = torch.zeros((1, sequence_len, d_model), dtype=torch.float32).to(self.device)
        factor = -math.log(10000.0) / d_model  # outs loop
        for index in range(0, sequence_len):  # position of word in seq
            for i in range(0, d_model, 2):
                div_term = math.exp(i * factor)
                pe[0, index, i] = math.sin(index * div_term)
                if(i+1<d_model):
                    pe[0, index, i+1] = math.cos(index * div_term)
                
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

    


"""
    cnn_transformer: A class of the CNN_transformer small model.
    Used for 101 velocity channels data (data with 101 length).
    
    input Attributes
    ----------
    num_output: int
        number of output
    in_channels: int
        number of input channels
    input_row : int 
        number of input row
    input_column: int
        number of input column
    drop_out_rate : float
        the drop out rate in last layer of CNN.
    lpe: boolean
        whether or not add leanable positional embedding to CNN inputs.
        
    Methods
    -------
    forward(x):
        calculate the outputs.
    
"""


class cnn_transformer(nn.Module):
    def __init__(self, num_output=2, in_channels=1, input_row = 2, input_column=101, drop_out_rate=0, lpe=False):
        super(cnn_transformer, self).__init__()
        
        self.num_features = 54
        self.input_row = input_row
        self.in_channels = in_channels
        self.input_column=input_column
        self.drop_rate=drop_out_rate
        self.lpe=lpe
        # useless
        self.pos_encoder = PositionalEncoding(num_features=self.num_features, sequence_len=6, d_model=9)
        
        self.pos_embedding = nn.Parameter(torch.randn(self.in_channels,self.input_row, self.input_column))
        # CNN layers
        if(input_row>=2):
            self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=72, 
                               kernel_size=(2,3), stride=1, padding=0, 
                               bias=True, padding_mode='zeros')
        else:
            self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=72, 
                               kernel_size=(1,3), stride=1, padding=0, 
                               bias=True, padding_mode='zeros')
            
        self.bn1 = nn.BatchNorm2d(72)
        
        self.conv2 = nn.Conv2d(in_channels= 72, out_channels=64,
                               kernel_size=(1,10), stride=1, padding=0, bias=True,
                               padding_mode='zeros')
        
        self.bn2 = nn.BatchNorm2d(64)
            
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels=56,
                               kernel_size=(1,3), stride=1, padding=0, bias=True, 
                               padding_mode='zeros')
        
        self.bn3 = nn.BatchNorm2d(56)
        
        self.conv4 = nn.Conv2d(in_channels= 56, out_channels=48, 
                               kernel_size=(1,10), stride=1, padding=0, bias=True,
                               padding_mode='zeros')
        
        self.bn4 = nn.BatchNorm2d(48)
        
        self.conv5 = nn.Conv2d(in_channels = 48, out_channels=40, 
                               kernel_size=(1,3), stride=1, padding=0,
                               bias=True, padding_mode='zeros')
        
        self.bn5 = nn.BatchNorm2d(40)
        
        self.conv6 = nn.Conv2d(in_channels= 40, out_channels=32, 
                               kernel_size=(1,10), stride=1,
                               padding=0, bias=True, padding_mode='zeros')
        
        self.bn6 = nn.BatchNorm2d(32)
        
        self.conv7 = nn.Conv2d(in_channels = 32, out_channels=16, 
                               kernel_size=(1,3), stride=1, padding=0, 
                               bias=True, padding_mode='zeros')
        
        self.bn7 = nn.BatchNorm2d(16)
        
        self.conv8 = nn.Conv2d(in_channels= 16, out_channels=8, 
                               kernel_size=(1,10), stride=1, 
                               padding=0, bias=True, padding_mode='zeros')
        
        self.bn8 = nn.BatchNorm2d(8)
        if(input_row<=2):
            self.linear = nn.Linear(456, 54)
        else:
            self.linear = nn.Linear(int(1904*(input_row-1)), 54)
            
        self.flatten = nn.Flatten()
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=9,
                nhead=3,
                dim_feedforward=36,
                dropout=self.drop_rate,
                batch_first=True,
            ),
            num_layers=4
        )
        
        self.decoder = nn.Linear(54, num_output)

        # init parameter
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if(self.lpe==True):
            x =  x + self.pos_embedding
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        #print(2, x.size())
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        #print(3, x.size())
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        #
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        #
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        #
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        #
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        #
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        #
        x = self.flatten(x)
        x = self.linear(x)
        #print(3, x.size())
        x = x.reshape(x.shape[0], -1, 9)
        # Transformer MODEL
        x = self.transformer(x)
        x = self.flatten(x)
        x = self.decoder(x)
        return x
    



    
    
"""
    cnn: A class of the CNN model.
    for 101 velocity channels. 1D Data with 101 length.
    
    input Attributes
    ----------
    
    num_output: int
        number of output
    in_channels: int
        number of input channels
    input_row : int 
        number of input row
    input_column: int
        number of input column
    num_layer : int
        number of layers of CNN will be 8. Users can not change this settings. 
    drop_out_rate : float
        the drop out rate in last layer of CNN.
    lpe: boolean
        whether or not add leanable positional embedding to CNN inputs.
    
    Methods
    -------
    forward(x):
        calculate the outputs.
    
"""
    
class cnn(nn.Module):
    def __init__(self, num_output=2, in_channels=1, input_row = 2, input_column=101, num_layer = 8, drop_out_rate=0.30, lpe=False):
        super(cnn, self).__init__()
        
        self.num_features = in_channels*input_row*input_column
        self.drop_rate=drop_out_rate
        self.lpe=lpe
        self.out_channels_1 = num_layer*8+8
        self.dropout = nn.Dropout(drop_out_rate)
        self.pos_embedding = nn.Parameter(torch.randn(in_channels, input_row, input_column))
        # head of CNN
        if(input_row>=2):
            self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=self.out_channels_1, 
                               kernel_size=(2,3), stride=1, padding=0, 
                               bias=True, padding_mode='zeros')
        else:
            self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=self.out_channels_1, 
                               kernel_size=(1,3), stride=1, padding=0, 
                               bias=True, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(self.out_channels_1)
        
        self.conv2 = nn.Conv2d(in_channels= self.out_channels_1, out_channels=self.out_channels_1-8,
                               kernel_size=(1,10), stride=1, padding=0, bias=True,
                               padding_mode='zeros')
        
        self.bn2 = nn.BatchNorm2d(self.out_channels_1-8)
        # add layers 
        self.layers = nn.ModuleList()
        self.layers.append(self.conv1)
        self.layers.append(self.bn1)
        self.layers.append(nn.ReLU())
        
        self.layers.append(self.conv2)
        self.layers.append(self.bn2)
        self.layers.append(nn.ReLU())
        count = self.out_channels_1-8
        if(num_layer>=4):
            for i in range(0, int((num_layer-4)/2)):
                #1
                self.layers.append(nn.Conv2d(in_channels= count, out_channels=count-8,
                               kernel_size=(1,3), stride=1, padding=0, bias=True,
                               padding_mode='zeros'))
                self.layers.append(nn.BatchNorm2d(count-8))
                self.layers.append(nn.ReLU())
                # 2
                self.layers.append(nn.Conv2d(in_channels= count-8, out_channels=count-16,
                               kernel_size=(1,10), stride=1, padding=0, bias=True,
                               padding_mode='zeros'))
                self.layers.append(nn.BatchNorm2d(count-16))
                self.layers.append(nn.ReLU())
                count = count-16
        ### count = 32
        if(num_layer>=4):
            self.layers.append(nn.Conv2d(in_channels = count, out_channels=16, 
                               kernel_size=(1,3), stride=1, padding=0, 
                               bias=True, padding_mode='zeros'))
        
            self.layers.append(nn.BatchNorm2d(16))
            self.layers.append(nn.ReLU())
        
            self.layers.append(nn.Conv2d(in_channels= 16, out_channels=8, 
                               kernel_size=(1,10), stride=1, 
                               padding=0, bias=True, padding_mode='zeros'))
            self.layers.append(nn.BatchNorm2d(8))
            self.layers.append(nn.ReLU())
        
        ###
        self.layers.append(nn.Flatten())
        self.layers.append(self.dropout)
        
        if(num_layer==2):
            self.linear = nn.Linear(1440, num_output)
        else:
            input_channels = 632 - 88 * int((num_layer-4)/2)
            self.linear = nn.Linear(input_channels, num_output)
        
        # init parameter
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if(self.lpe==True):
            x = x + self.pos_embedding
        for layer in self.layers[:-1]:
            #print('layer=', layer)
            x = layer(x)
        x = self.linear(x)
        return x

    
    