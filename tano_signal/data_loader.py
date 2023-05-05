# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 21:42:38 2023

@author: Administrator-1
"""
import numpy as np
import torch
from torch.utils.data import Dataset

""" 
    spectra_loader: A class to load the data.
    This loader load the data in the common way.

    Attributes
    ----------
    x : tensor
        training data 
    y : tensor
        lebels
    transform : set
        set of transformations for x.
    target_transform : set
        set of transformations for y.
    mode: Str
        Data representations.
    num_column: int
        number of columns for input data.

    Methods
    -------
    __getitem__(index):
        get the data in the index. it will choose the loading methods based on the mode.
"""


# data loader 
class loader(Dataset):
    def __init__(self, x, y, x_transform=None, y_transform=None, PEV=None):
        
        self.x = x
        self.y = y
        self.transform = x_transform
        self.target_transform = y_transform
        self.mode = PEV
        self.num_column = x.shape[1]
    def __len__(self):
        return len(self.y[:,0])
    
    def __getitem__(self, idx):
        spectra = self.x[idx,:]
        label = self.y[idx,:]
            
        if (self.mode=='index_c'):
            position = np.linspace(0, 1.0, self.num_column).reshape(1, -1)
            spectra = np.vstack((spectra, position))
            spectra = spectra.reshape(1, spectra.shape[0], spectra.shape[1])
        elif (self.mode=='index_a'):
            position = np.linspace(0, 1.0, self.num_column).reshape(1, -1)
            spectra = spectra + position
            spectra = spectra.reshape(1,1,spectra.shape[1])
        elif (self.mode=='sin_a'):
            position = np.sin(np.linspace(0, 1.0, self.num_column).reshape(1, -1))
            spectra = spectra + position
            spectra = spectra.reshape(1,1,spectra.shape[1])
        elif (self.mode=='sin_c'):
            position = np.sin(np.linspace(0, 1.0, self.num_column).reshape(1, -1))
            spectra =  np.vstack((spectra,position))
            spectra = spectra.reshape(1, spectra.shape[0], spectra.shape[1])
        elif (self.mode=='poly_c'):
            zero = np.linspace(0, 1.0, self.num_column).reshape(1, -1) 
            two =  spectra**2
            spectra = np.vstack((spectra,two))
            spectra = spectra.reshape(1, spectra.shape[0], spectra.shape[1])
        else:
            spectra = spectra.reshape(1,1,spectra.shape[0])
            
        if self.transform:
            spectra = self.transform(spectra)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return spectra, label

    
""" 
    spectra_cube_loader: A class to load the datacube.

    Attributes
    ----------
    xy : dictionary with form of [key: value]. value = (spectra, coordinates(x, y), ground truth) 
        training data 
    transform : set
        set of transformations for x.
    target_transform : set
        set of transformations for y.
    mode: Str
        Data representations.
    num_column: int
        number of columns for input data.

    Methods
    -------
    __getitem__(index):
        get the data in the index. it will choose the loading methods based on the mode.
"""


# data loader 
class cube_loader(Dataset):
    def __init__(self, data_dict, x_transform=None, y_transform=None, PEV=None):
        
        self.xy = data_dict
        self.transform = x_transform
        self.target_transform = y_transform
        self.mode=PEV
        self.num_column = len(self.xy[0][0])
    def __len__(self):
        return len(self.xy)
    
    def __getitem__(self, idx):
        spectra = self.xy[idx][0]
        label = self.xy[idx][2]
        
        if (self.mode=='index_c'):
            position = np.linspace(0, 1.0, self.num_column).reshape(1, -1)
            spectra = np.vstack((spectra, position))
            spectra = spectra.reshape(1, spectra.shape[0], spectra.shape[1])
        elif (self.mode=='index_a'):
            position = np.linspace(0, 1.0, self.num_column).reshape(1, -1)
            spectra = spectra + position
            spectra = spectra.reshape(1,1,spectra.shape[1])
        elif (self.mode=='sin_a'):
            position = np.sin(np.linspace(0, 1.0, self.num_column).reshape(1, -1))
            spectra = spectra + position
            spectra = spectra.reshape(1,1,spectra.shape[1])
        elif (self.mode=='sin_c'):
            position = np.sin(np.linspace(0, 1.0, self.num_column).reshape(1, -1))
            spectra =  np.vstack((spectra,position))
            spectra = spectra.reshape(1, spectra.shape[0], spectra.shape[1])
        elif (self.mode=='poly_c'):
            two =  spectra**2
            spectra = np.vstack((spectra,two))
            spectra = spectra.reshape(1, spectra.shape[0], spectra.shape[1])
        else:
            spectra = spectra.reshape(1,1,spectra.shape[0])
            
        if self.transform:
            spectra = np.array(spectra).astype(np.float32)
            spectra = self.transform(spectra)
            
        if self.target_transform:
            label = np.array(label).astype(np.float32)
            label = self.target_transform(label)
            
        return spectra, label
    

    
class ToTensor():
    def __call__(self, sample):
        x = torch.from_numpy(sample)
        return x
    

    
    
