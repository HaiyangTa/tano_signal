
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
from lib import preprocesser 


def getitem(spectra, PEV):
    spectra = preprocesser.preprocess(spectra, PEV)
    return spectra
        
    
def calculate(dictt, model, device, num_row, num_column, PEV):
    model.eval()
    Fcnm = np.zeros((num_row, num_column,1))
    Rhi = np.zeros((num_row,num_column,1))
    for i in range(0, len(dictt)):
        if(i%10000==0):
            print(i)
        o = dictt.get(i)

        spectra = getitem(spectra = np.array(o[0]), PEV=PEV)
        #print('spectra1=',spectra.shape)
        spectra = spectra.reshape(1,spectra.shape[0], spectra.shape[1], -1).astype(np.float32)
        #print('spectra2=',spectra.shape)
        spectra = torch.from_numpy(spectra)
        spectra = spectra.to(device)
        #print('spectra=',spectra)
        position = o[1]
        #print('position=', position)
        prediction = model(spectra)
        Fcnm[position[0], position[1]] =prediction[0][0].cpu().detach().numpy()
        Rhi[position[0], position[1]] =prediction[0][1].cpu().detach().numpy()
    return Fcnm, Rhi