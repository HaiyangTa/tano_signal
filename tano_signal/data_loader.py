# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 21:42:38 2023

@author: Haiyang Tang
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from  tano_signal.data_changer import dictionarizer, preprocessor


""" 
    spectra_loader: A class to load the data.
    This loader load the data in the common way.

    Attributes
    ----------
    x : tensor
        training data 
    y : tensor
        lebels
    x_transform : set
        set of transformations for x.
    y_transform : set
        set of transformations for y.
    PEV: Str
        Data representations.

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
            
        spectra = spectra = preprocessor.preprocess(spectra = spectra, PEV = self.mode)
        spectra = np.array(spectra).astype(np.float32)
            
        if self.transform:
            spectra = self.transform(spectra)
        spectra = torch.from_numpy(spectra)
        
        if self.target_transform:
            label = self.target_transform(label)
        label = torch.from_numpy(label)
        
        return spectra, label

    
""" 
    spectra_cube_loader: A class to load the datacube.

    Attributes
    ----------
    xy : dictionary with form of [key: value]. value = (spectra, coordinates(x, y), ground truth) 
        training data 
    x_transform : set
        set of transformations for x.
    y_transform : set
        set of transformations for y.
    PEV: Str
        Data representations.

    Methods
    -------
    __getitem__(index):
        get the data in the index. it will choose the loading methods based on the mode.
"""


# data loader 
class cube_loader(Dataset):
    def __init__(self, cube, fcnm, rhi , x_transform=None, y_transform=None, PEV=None):
        
        self.cube = cube
        self.rhi = rhi
        self.fcnm = fcnm
        self.transform = x_transform
        self.target_transform = y_transform
        self.dm = dictionarizer.dict_maker(cube = self.cube ,Rhi=self.rhi, Fcnm=self.fcnm)
        self.xy = self.dm.make_dict_gt()
        self.mode=PEV
        self.num_column = len(self.xy[0][0])
        
        
    def __len__(self):
        return len(self.xy)
    
    def __getitem__(self, idx):
        spectra = self.xy[idx][0]
        label = self.xy[idx][2]
        spectra = preprocessor.preprocess(spectra = spectra, PEV = self.mode)
        spectra = np.array(spectra).astype(np.float32)
        
        
        if self.transform:
            spectra = self.transform(spectra)
        spectra = torch.from_numpy(spectra)
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return spectra, label
    

    
    

    
    
