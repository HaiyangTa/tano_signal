
"""
@author: Haiyang.Tang
Do not used for commercial purpose!
"""

import numpy as np
import math

"""
    Dictionarizer_gt: A class of dictionarizer.
    generate a dictionary for the datacube that conatins ground truth. 
    Used for 101 velocity channels data (data with 101 length).
    
    input Attributes
    ----------
    num_row : int
        The number of row.
    num_column : int.
        input column number.
    num_channel : int.
        number of channels
    cube: data cube
        spectra data cube.
    Rhi: cube 
        Rhi ground truth
    Fcnm:  cube
        Fcnm ground truth.
        
    Methods
    -------
    dictionarize:
        generate a dictionary for the datacube that conatins ground truth. 
    
"""


class Dictionarizer_gt:

    def __init__(self, num_row, num_column, num_channel, cube ,Rhi, Fcnm):
        self.num_row = num_row
        self.num_column = num_column
        self.num_channel = num_channel
        self.num = num_row* num_column
        self.Rhi = Rhi
        self.Fcnm = Fcnm
        self.dictt = {}
        self.cube = cube
        
    def dictionarize(self):
        for i in range(0, self.num):
            index=i
            row_index = index//self.num_column
            column_index = index%self.num_column
            self.dictt[i] = (self.cube[:,row_index, column_index],
                             [self.Fcnm[row_index, column_index], self.Rhi[row_index, column_index]],
                             [row_index, column_index])
        return self.dictt
    
    
"""
    Dictionarizer: A class of dictionarizer.
    generate a dictionary for the datacube that dose not conatin ground truth. 
    Used for 101 velocity channels data (data with 101 length).
    
    input Attributes
    ----------
    num_row : int
        The number of row.
    num_column : int.
        input column number.
    num_channel : int.
        number of channels
    cube: data cube
        spectra data cube.
        
    Methods
    -------
    dictionarize:
        generate a dictionary for the datacube that not conatins ground truth. 
    
"""
    
    
class Dictionarizer:

    def __init__(self, num_row, num_column, num_channel, cube):


        self.dictt = {}
        self.cube = cube
        self.num_row = num_row
        self.num_column = num_column
        self.num_channel = num_channel
        self.num = num_row* num_column
        
    def dictionarize(self):
        for i in range(0, self.num):
            index=i
            row_index = index//self.num_column
            column_index = index%self.num_column
            self.dictt[i] = (self.cube[:,row_index, column_index],
                             [row_index, column_index])
        return self.dictt