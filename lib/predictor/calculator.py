



import numpy as np
import torch
from torch.utils.data import Dataset
#from lib import preprocesser
from lib.data_changer import dictionarizer
from lib.data_changer import preprocessor


def getitem(spectra, PEV):
    spectra = preprocessor.preprocess(spectra, PEV)
    return spectra



"""
    calculate: A method to predict the RHI and FCNM.
    
    input Attributes
    ----------
    cube: data cube 
        the 3D spectra
    model: model 
        model for prediction. 
    PEV: string
        positional encoding vector. 
        
"""
    
def calculate(cube, model, PEV):
    model.eval()
    num_row = cube.shape[1]
    num_column = cube.shape[2]
    dictt = dictionarizer.dict_maker(cube=cube,Rhi=None, Fcnm=None).make_dict()
    
    device = torch.device('cpu')
    model.to(device)
    
    Fcnm = np.zeros((num_row, num_column,1))
    Rhi = np.zeros((num_row,num_column,1))
    for i in range(0, len(dictt)):
        if(i%10000==0):
            print(i)
        o = dictt.get(i)

        spectra = getitem(spectra = np.array(o[0]), PEV=PEV)
        spectra = spectra.reshape(1,spectra.shape[0], spectra.shape[1], -1).astype(np.float32)
        spectra = torch.from_numpy(spectra)
        spectra = spectra.to(device)
        position = o[1]
        #print('position=', position)
        prediction = model(spectra)
        Fcnm[position[0], position[1]] =prediction[0][0].cpu().detach().numpy()
        Rhi[position[0], position[1]] =prediction[0][1].cpu().detach().numpy()
    return Fcnm, Rhi