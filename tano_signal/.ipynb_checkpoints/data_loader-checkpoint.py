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
class spectra_loader(Dataset):
    def __init__(self, x, y, transform=None, target_transform=None, pe=None):
        
        self.x = x
        self.y = y
        self.transform = transform
        self.target_transform = target_transform
        self.mode=pe
        self.num_column = x.shape[1]
    def __len__(self):
        return len(self.y[:,0])
    
    def __getitem__(self, idx):
        spectra = self.x[idx,:]
        label = self.y[idx,:]
            
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
            
        if self.transform:
            spectra = self.transform(spectra)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return spectra, label
    

    

""" 
    spectra_loader_alte: A class to load the data with the alter way.
    
    This loader only deal with representations that number of columns >= 2. 
    It loads the data in the alternative way. For example:
    x= [[1, 2, 3],
       [0.1, 0.2, 0.3]]
    This loader will give you :
    x = [1, 0.1, 2, 0.2, 3, 0.3]
    The reason to apply this loading method is to deal with the reshape of the transformer.
    In this case, x will be reshaped to 
    x = [[1, 0.1], [2, 0.2], [3, 0.3]]
    where each values is combined with the positional encoding. If we use the original methods, x will be
    x = [[1, 2],[3, 0.1],[0.2,0.3]], which makes no sense.
    
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
class spectra_loader_alte(Dataset):
    def __init__(self, x, y, transform=None, target_transform=None, pe=None):
        
        self.x = x
        self.y = y
        self.transform = transform
        self.target_transform = target_transform
        self.mode=pe
        self.num_column = x.shape[1]
    def __len__(self):
        return len(self.y[:,0])
    
    def __getitem__(self, idx):
        spectra = self.x[idx,:]
        label = self.y[idx,:]
        if (self.mode=='index_concate'):
            position = np.linspace(0, 1.0, self.num_column).reshape(1, -1)
            if self.transform:
                c=[]
                for i in range(len(spectra)):
                    c.append(spectra[i])
                    c.append(position[0,i])
                spectra = np.array(c).astype(np.float64)
                spectra = spectra.reshape(1, 1,spectra.shape[0])
        elif (self.mode=='sin_concate'):
            position = np.sin(np.linspace(0, 1.0, self.num_column).reshape(1, -1))
            if self.transform:
                c=[]
                for i in range(len(spectra)):
                    c.append(spectra[i])
                    c.append(position[0,i])
                spectra = np.array(c).astype(np.float64)
                spectra = spectra.reshape(1, 1,spectra.shape[0])
                
        elif (self.mode=='poly_concate'):
            zero = np.linspace(0, 1.0, 414)
            two =  spectra**2
            label = self.y[idx,:]
            if self.transform:
                c=[]
                for i in range(len(spectra)):
                    c.append(zero[i])
                    c.append(spectra[i])
                    c.append(two[i])
                spectra = np.array(c).astype(np.float64)
                spectra = spectra.reshape(1, 1,spectra.shape[0])
        else:
            print("warning: No such pe!")
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
    xy : tuple object
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
class spectra_cube_loader(Dataset):
    def __init__(self, xy, transform=None, target_transform=None, pe=None):
        
        self.xy = xy
        self.transform = transform
        self.target_transform = target_transform
        self.mode=pe
        self.num_column = len(self.xy[0][0])
    def __len__(self):
        return len(self.xy)
    
    def __getitem__(self, idx):
        spectra = self.xy[idx][0]
        label = self.xy[idx][1]
        
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
    

    
    
