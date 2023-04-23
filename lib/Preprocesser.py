# -*- coding: utf-8 -*-
"""

@author: Haiyang.Tang
"""
import numpy as np

""" 
    proprocess: preprocess data using different PEV.
    This loader load the data in the common way.

    Attributes
    ----------
    spectra : 1D spectra data 
        training data 
    PEV : string
         positional encoding method

    return 
    -------
    spectra: np array
        spectra after processing.
"""

def preprocess(spectra, PEV):
    #print('shape=', spectra.shape)
    num_column = len(spectra)
    
    
    if (PEV=='index_concate'):
        position = np.linspace(0, 1.0, num_column).reshape(1, -1)
        spectra = np.vstack((spectra, position))
        spectra = spectra.reshape(1, spectra.shape[0], spectra.shape[1])
    elif (PEV=='index_add'):
        position = np.linspace(0, 1.0, num_column).reshape(1, -1)
        spectra = spectra + position
        spectra = spectra.reshape(1,1,spectra.shape[1])
    elif (PEV=='sin_add'):
        position = np.sin(np.linspace(0, 1.0, num_column).reshape(1, -1))
        spectra = spectra + position
        spectra = spectra.reshape(1,1,spectra.shape[1])
    elif (PEV=='sin_concate'):
        position = np.sin(np.linspace(0, 1.0, num_column).reshape(1, -1))
        spectra =  np.vstack((spectra,position))
        spectra = spectra.reshape(1, spectra.shape[0], spectra.shape[1])
    elif (PEV=='poly_concate'):
        zero = np.linspace(0, 1.0, num_column).reshape(1, -1) 
        two =  spectra**2
        spectra = np.vstack((spectra,two))
        spectra = spectra.reshape(1, spectra.shape[0], spectra.shape[1])
    else:
        spectra = spectra.reshape(1,1,spectra.shape[0])
        
    return spectra