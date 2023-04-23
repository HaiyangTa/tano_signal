
from lib import prototype as prototype
import torch
import os
#import model.cnn_transformer as cnn_transformer
#import model.cnn as cnn 

"""
    spectra_cnn_transformerr: get the model of CNN-transformer.
    Used for 101 velocity channels data (data with 101 length).
    
    input Attributes
    ----------
    num_output : int
        The number of features.
    in_channels : int. 
        input channel number.
    input_row : int.
        intput row number.
    input_column: int
        input columsn number.
    num_layer: int.
        number of layers.
    drop_out_rate: float 
        drop out rate in the last layer.
    lpe: boolean
        learnable positional encoding.
    weights:weights
        weights for the model.
    
    return:
    -------
    modell:
        return the model. 
    
"""

def spectra_cnn_transformer(weights, num_output=2, in_channels=1, input_row = 1, input_column=101, drop_out_rate=0, lpe=False):
    
    modell = prototype.cnn_transformer(num_output= num_output, 
                                             in_channels=in_channels, 
                                             input_row = input_row, 
                                             input_column=input_column, 
                                             drop_out_rate=drop_out_rate, 
                                             lpe=lpe)
    
    if(weights != None): 
        modell.load_state_dict(weights.get_checkpoint_weights())
    return modell
        
        

"""
    learnable_PEV_ct_weights,: A class of the CNN_transformer weights (learnable_PEV_ct_weights).
    Used for 101 velocity channels data (data with 101 length).
    
    input Attributes
    ----------
    device : device
        GPU or CPU.
    
    Methods
    -------
    get_checkpoint():
        return the checkpoint of the weights. 
    
"""
        
        
class learnable_PEV_ct_weights:
    def __init__(self, device):
        self.path =  os.getcwd() + '\\lib\\learnable_PEV.pth'
        print(self.path)
        self.checkpoint = torch.load(self.path, map_location=device)
        
    def get_checkpoint_weights(self):
        
        return self.checkpoint['net']

    
"""
    spectra_cnn: A class of the CNN small model.
    Used for 101 velocity channels data (data with 101 length).
    
    input Attributes
    ----------
    num_output : int
        The number of features.
    in_channels : int. 
        input channel number.
    input_row : int.
        intput row number.
    input_column: int
        input columsn number.
    num_layer: int.
        number of layers.
    drop_out_rate: float 
        drop out rate in the last layer.
    lpe: boolean
        learnable positional encoding.
    weights:weights
        weights for the model.
    
    Methods
    -------
    __init__():
        return the model. 
    
"""



def spectra_cnn(weights, num_output,in_channels,input_row, input_column, drop_out_rate, lpe):
    
    
    modell = prototype.cnn(num_output= num_output,
                                             in_channels=in_channels,
                                             input_row = input_row,
                                             input_column=input_column,
                                             drop_out_rate=drop_out_rate, 
                                             lpe=lpe)
        
    if(weights != None):
        modell.load_state_dict(weights.get_checkpoint_weights())
    
    return modell
        
        

"""
    learnable_PEV_ct_weights: A class of the CNN_transformer weights (learnable_PEV_ct_weights).
    Used for 101 velocity channels data (data with 101 length).
    
    input Attributes
    ----------
    device : device
        GPU or CPU.
    
    Methods
    -------
    get_checkpoint():
        return the checkpoint of the weights. 
    
"""
        
        
class poly_concate_c_weights:
    
    def __init__(self, device):
        self.path =  os.getcwd() + '\\lib\\poly_concate.pth'
        print(self.path)
        self.checkpoint = torch.load(self.path, map_location=device)
        
    def get_checkpoint_weights(self):
        
        return self.checkpoint['net']