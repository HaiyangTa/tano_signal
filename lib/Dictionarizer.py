
"""
@author: Haiyang.Tang
Do not used for commercial purpose!
"""

import numpy as np
import math

"""
    dict_maker_gt: A class of dict_maker.
    generate a dictionary for the datacube that conatins ground truth. 
    Used for 101 velocity channels data (data with 101 length).
    
    input Attributes
    ----------
    cube: data cube
        spectra data cube.
    Rhi: cube 
        Rhi ground truth
    Fcnm:  cube
        Fcnm ground truth.
        
    Methods
    -------
    make_dict:
        generate a dictionary for the datacube that conatins ground truth. 
    
"""


class dict_maker_gt:

    def __init__(self, cube ,Rhi, Fcnm):
        
        self.Rhi = Rhi
        self.Fcnm = Fcnm
        self.dictt = {}
        self.cube = cube
        self.num_row = self.cube.shape[1]
        self.num_column = self.cube.shape[2]
        self.num_channel = self.cube.shape[0]
        self.num = self.num_row* self.num_column
        
    def make_dict(self):
        for i in range(0, self.num):
            index=i
            row_index = index//self.num_column
            column_index = index%self.num_column
            self.dictt[i] = (self.cube[:,row_index, column_index],
                             [row_index, column_index],
                             [self.Fcnm[row_index, column_index], self.Rhi[row_index, column_index]]
                             )
        return self.dictt
    
    
"""
    dict_maker: A class of dict_maker.
    generate a dictionary for the datacube that dose not conatin ground truth. 
    Used for 101 velocity channels data (data with 101 length).
    
    input Attributes
    ----------
    cube: data cube
        spectra data cube.
        
    Methods
    -------
    make_dict:
        generate a dictionary for the datacube that not conatins ground truth. 
    
"""
    
    
class dict_maker:

    def __init__(self, cube):
        self.dictt = {}
        self.cube = cube
        self.num_row = self.cube.shape[1]
        self.num_column = self.cube.shape[2]
        self.num_channel = self.cube.shape[0]
        self.num = self.num_row* self.num_column
        
    def make_dict(self):
        for i in range(0, self.num):
            index=i
            row_index = index//self.num_column
            column_index = index%self.num_column
            self.dictt[i] = (self.cube[:,row_index, column_index],
                             [row_index, column_index])
        return self.dictt