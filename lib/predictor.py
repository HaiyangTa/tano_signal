
"""
    predict: A method to predict the RHI and FCNM.
    
    input Attributes
    ----------
    dictt: dictionary 
        the  dictionary that contains only one data cube spectra.
    model: model 
        model for prediction. 
    device: GPU or CPU 
        the device for calculation.
    num_row: int 
        number of row.
    num_column: int
        number of column.
"""


import numpy as np
import torch
from torch.utils.data import Dataset



class predictor:
    
    def __init__(self, dictt, model, device, num_row, num_column, PEV):
        
        self.model = model
        self.device = device
        self.dictt = dictt
        self.num_row = num_row
        self.num_column = num_column
        self.mode = PEV
        
    def getitem(self, spectra):
            
        if (self.mode=='index_concate'):
            position = np.linspace(0, 1.0, self.num_column).reshape(1, -1)
            spectra = np.vstack((spectra, position))
            spectra = spectra.reshape(1, spectra.shape[0], spectra.shape[1])
        elif (self.mode=='index_add'):
            position = np.linspace(0, 1.0, self.num_column).reshape(1, -1)
            spectra = spectra + position
            spectra = spectra.reshape(1,1,spectra.shape[1])
        elif (self.mode=='sin_add'):
            position = np.sin(np.linspace(0, 1.0, self.num_column).reshape(1, -1))
            spectra = spectra + position
            spectra = spectra.reshape(1,1,spectra.shape[1])
        elif (self.mode=='sin_concate'):
            position = np.sin(np.linspace(0, 1.0, self.num_column).reshape(1, -1))
            spectra =  np.vstack((spectra,position))
            spectra = spectra.reshape(1, spectra.shape[0], spectra.shape[1])
        elif (self.mode=='poly_concate'):
            zero = np.linspace(0, 1.0, self.num_column).reshape(1, -1) 
            two =  spectra**2
            spectra = np.vstack((spectra,two))
            spectra = spectra.reshape(1, spectra.shape[0], spectra.shape[1])
        else:
            spectra = spectra.reshape(1,1,spectra.shape[0])
            
        return spectra
        
    
    def predict():
        Fcnm = np.zeros((self.num_row, self.num_column,1))
        Rhi = np.zeros((self.num_row,self.num_column,1))
        for i in range(0, len(self.dictt)):
            o = dictt.get(i)
            spectra = getitem(np.array(o[0]))
            print('shape = ', spectra.shape)
            spectra = spectra.reshape(1,spectra.shape[0], spectra.shape[1],spectra.shape[2]).astype(np.float32)
            spectra = torch.from_numpy(spectra)
            spectra = spectra.to(self.device)
            position = o[1]
            prediction = self.model(spectra)
            Fcnm[position[0], position[1]] =abs(prediction[0][0].cpu()).detach().numpy()
            Rhi[position[0], position[1]] =abs(prediction[0][1].cpu()).detach().numpy()
        return Fcnm, Rhi