
from tano_signal.models import prototype
import torch
import os
import importlib 

"""
    spectra_cnn_transformerr: get the model of CNN-transformer.
    Used for 101 velocity channels data (data with 101 length).
    
    input Attributes
    ----------
    num_output : int
        The number of features.
    drop_out_rate: float 
        drop out rate in the last layer.
    PEV: stinrg
        positional encoding vector.
    weights:weights
        weights for the model.
    
    return:
    -------
    modell:
        return the model. 
    
"""

def spectra_cnn_transformer(PEV, weights, num_output=2, drop_out_rate=0):
    
    input_column = 101
    in_channels = 1
    input_row = 1
    lpe = False
    
    
    if(PEV=='trainable_a'):
        lpe=True
    if (PEV=='index_c'):
        input_row = 2
    elif (PEV=='sin_c'):
        input_row = 2
    elif (PEV=='poly_c'):
        input_row = 2
    modell = prototype.cnn_transformer(num_output= num_output, 
                                             in_channels=in_channels, 
                                             input_row = input_row, 
                                             input_column=input_column, 
                                             drop_out_rate=drop_out_rate, 
                                             lpe=lpe)
    
    if(weights != None): 
        modell.load_state_dict(weights.get_checkpoint_weights())
        print('sucessfully load the weights!')
    modell.eval()
    
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
    get_checkpoint_weights():
        return the checkpoint of the weights. 
    
"""
        
        
class learnable_PEV_ct_weights:
    def __init__(self, device):
        with importlib.resources.path('tano_signal.models', 'learnable_PEV.pth') as path:
            self.path = path
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
    drop_out_rate: float 
        drop out rate in the last layer.
    PEV: stinrg
        positional encoding vector.
    weights:weights
        weights for the model.
    
    return 
    -----------
    spectra_cnn model object

    
"""



def spectra_cnn(PEV, weights, num_output=2, drop_out_rate=0):
    
    input_column = 101
    in_channels = 1
    input_row = 1
    lpe = False
    
    if(PEV=='trainable_a'):
        lpe=True
    if (PEV=='index_c'):
        input_row = 2
    elif (PEV=='sin_c'):
        input_row = 2
    elif (PEV=='poly_c'):
        input_row = 2
        
    
    modell = prototype.cnn(num_output= num_output,
                                             in_channels=in_channels,
                                             input_row = input_row,
                                             input_column=input_column,
                                             drop_out_rate=drop_out_rate, 
                                             lpe=lpe)
        
    if(weights != None):
        modell.load_state_dict(weights.get_checkpoint_weights())
        print('sucessfully load the weights!')
    modell.eval()
    
    return modell
        
        

"""
    poly_concate_c_weights: A class of the CNN weights (poly_concate_c_weights).
    Used for 101 velocity channels data (data with 101 length).
    
    input Attributes
    ----------
    device : device
        GPU or CPU.
    
    
    Methods
    -------
    get_checkpoint_weights():
        return the checkpoint of the weights. 
    
"""


class poly_concate_c_weights:
    
    def __init__(self, device):
        with importlib.resources.path('tano_signal.models', 'poly_concate.pth') as path:
            self.path = path
        print(self.path)
        self.checkpoint = torch.load(self.path, map_location=device)
        
    def get_checkpoint_weights(self):
        return self.checkpoint['net']

    
"""
    original_vector_c_weights: A class of the CNN weights (original vector).
    Used for 101 velocity channels data (data with 101 length).
    
    input Attributes
    ----------
    device : device
        GPU or CPU.
    
    Methods
    -------
    get_checkpoint_weights():
        return the checkpoint of the weights. 
    
"""

    
class original_vector_c_weights:
    
    def __init__(self, device):
        with importlib.resources.path('tano_signal.models', 'original_vector.pth') as path:
            self.path = path
        print(self.path)
        self.checkpoint = torch.load(self.path, map_location=device)
        
    def get_checkpoint_weights(self):
        return self.checkpoint['net']