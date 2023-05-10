from __future__ import print_function, division
import unittest
from tano_signal import *
from astropy.io  import fits
import numpy as np
import torch
import matplotlib.pyplot as plt


"""
    data_changer_test: A class to test data changer.
    
    input Attributes
    ----------
    dictt : dictionray 
        dictionray with not ground truth.
    dictt_gt: dictionary 
        dictionray with ground truth.
    cube : array 
        cubic data
    fcnm: array
        Fcnm ground truth 
    rhi: array
        Rhi ground truth 
    
    Methods
    -------
    (1-6): test the preprocess.py 
    (7-9): test the dictionarizer.py
    
"""

class data_changer_test(unittest.TestCase):
    
    def __init__(self, dictt, dictt_gt, cube, fcnm, rhi):
        self.spectra = np.array([1,2,3,4,5,6,7,8])
        self.num_column = len(self.spectra)
        self.dictt = dictt
        self.dictt_gt = dictt_gt
        self.cube = cube
        self.fcnm = fcnm
        self.rhi = rhi
        self.dict_maker_gt = tano_signal.dict_maker(cube = self.cube ,Rhi = self.rhi, Fcnm=self.fcnm)
        self.dict_maker = tano_signal.dict_maker(cube = self.cube ,Rhi = None, Fcnm=None)
    
    
    def compare_tuple(self, t1, t2):
        for i in range(len(t1)):
            A = np.array(t1[i])
            B = np.array(t2[i])
            if (A == B).all()==False:
                return False
        return True
    
    def compare_dict(self, d1, d2):
        for i in range (len(d1)):
            y = self.compare_tuple(d1[i], d2[i])
            if y ==False: 
                return False
        return True
    
    def preprocess_test1(self):
        # index_c
        print('start 1')
        position = np.linspace(0, 1.0,self.num_column).reshape(1, -1)
        truth = np.vstack((self.spectra, position))
        truth = truth.reshape(1, truth.shape[0], truth.shape[1])
        result = tano_signal.preprocess(self.spectra, 'index_c') 
        assert (result==truth).all()
        print('pass 1')
        #self.spectra = [1,2,3,4,5,6,7,8]
        
        
    def preprocess_test2(self):
        # index_a
        print('start 2')
        position = np.linspace(0, 1.0, self.num_column).reshape(1, -1)
        truth = self.spectra + position
        truth = truth.reshape(1,1, truth.shape[1])
        result = tano_signal.preprocess(self.spectra, 'index_a') 
        assert (result==truth).all()
        print('pass 2')
        
        
    def preprocess_test3(self):
        # sin_a
        print('start 3')
        position = np.sin(np.linspace(0, 1.0, self.num_column).reshape(1, -1))
        truth = self.spectra + position
        truth = truth.reshape(1,1,truth.shape[1])
        result = tano_signal.preprocess(self.spectra, 'sin_a') 
        assert (result==truth).all()
        print('pass 3')
        
        
    def preprocess_test4(self):
        # sin_c
        print('start 4')
        position = np.sin(np.linspace(0, 1.0, self.num_column).reshape(1, -1))
        truth =  np.vstack((self.spectra,position))
        truth = truth.reshape(1, truth.shape[0], truth.shape[1])
        result = tano_signal.preprocess(self.spectra, 'sin_c') 
        assert (result==truth).all()
        print('pass 4')
        
        
    def preprocess_test5(self):
        # poly_c
        print('start 5')
        two =  (self.spectra)**2
        truth = np.vstack((self.spectra,two))
        truth = truth.reshape(1, truth.shape[0], truth.shape[1])
        result = tano_signal.preprocess(self.spectra, 'poly_c') 
        assert (result==truth).all()
        print('pass 5')
        
        
    def preprocess_test6(self):
        # None or orther
        print('start 6')
        result = tano_signal.preprocess(self.spectra, None)
        truth = self.spectra.reshape(1,1,self.spectra.shape[0])
        assert (result==truth).all()
        print('pass 6')
        

    def dictionarizer_test7(self):
        #dict with ground truth 
        print('start 7')
        dictt_gt = self.dict_maker_gt.make_dict_gt()
        assert self.compare_dict(dictt_gt, self.dictt_gt) ==True
        print('pass 7')
        
    def dictionarizer_test8(self):
        # no ground truth 
        print('start 8')
        dictt = self.dict_maker.make_dict()
        assert self.compare_dict(dictt, self.dictt) ==True
        print('pass 8')

        
    def dictionarizer_test9(self):
        # no ground truth -> ground truth 
        print('start 9')
        dictt = self.dict_maker.make_dict_gt()
        assert (dictt==None)
        print('pass 9')
        
        
        
"""
    data_loader_test: A class to test data loader.
    
    input Attributes
    ----------
    cube : array 
        cubic data
    fcnm: array
        Fcnm ground truth 
    rhi: array
        Rhi ground truth 
    bs: int
        batch size
    
    Methods
    -------
    (1-6): test the shape of data loader for each representations. 
    
"""
        
        
class data_loader_test(unittest.TestCase):
    
    def __init__(self, cube, fcnm, rhi, bs=20):
        self.fcnm = fcnm
        self.rhi = rhi
        self.cube = cube
        self.batchsize  = bs
        
    def loader_test1(self):
        # test index_c
        print('start 1')
        dataset = tano_signal.cube_loader(cube=self.cube, 
                                          fcnm=self.fcnm, 
                                          rhi=self.rhi, 
                                          x_transform=None, 
                                          y_transform=None, 
                                          PEV='index_c')
    
        loader = torch.utils.data.DataLoader(dataset = dataset,
                                           batch_size = self.batchsize,
                                           shuffle =True)
        
        v = iter(loader)
        spectra, label = next(v)
        assert (list(spectra.shape)== [20, 1, 2, 101])
        assert (len(label)== 2)
        assert (len(label[0]) == self.batchsize)
        print('pass 1')
        
        
    def loader_test2(self):
        # test index_a
        print('start 2')
        dataset = tano_signal.cube_loader(cube=self.cube, 
                                          fcnm=self.fcnm, 
                                          rhi=self.rhi, 
                                          x_transform=None, 
                                          y_transform=None, 
                                          PEV='index_a')
    
        loader = torch.utils.data.DataLoader(dataset = dataset,
                                           batch_size = self.batchsize,
                                           shuffle =True)
        
        
        v = iter(loader)
        spectra, label = next(v)
        assert (list(spectra.shape )== [20, 1, 1, 101])
        assert (len(label)== 2)
        assert (len(label[0]) == self.batchsize)
        print('pass 2')
        
    def loader_test3(self):
        # test sin_c
        print('start 3')
        dataset = tano_signal.cube_loader(cube=self.cube, 
                                          fcnm=self.fcnm, 
                                          rhi=self.rhi, 
                                          x_transform=None, 
                                          y_transform=None, 
                                          PEV='sin_c')
    
        loader = torch.utils.data.DataLoader(dataset = dataset,
                                           batch_size = self.batchsize,
                                           shuffle =True)
        
        v = iter(loader)
        spectra, label = next(v)
        assert (list(spectra.shape)== [20, 1, 2, 101])
        assert (len(label)== 2)
        assert (len(label[0]) == self.batchsize)
        print('pass 3')
        
    def loader_test4(self):
        # test sin_a
        print('start 4')
        dataset = tano_signal.cube_loader(cube=self.cube, 
                                          fcnm=self.fcnm, 
                                          rhi=self.rhi, 
                                          x_transform=None, 
                                          y_transform=None, 
                                          PEV='sin_a')
    
        loader = torch.utils.data.DataLoader(dataset = dataset,
                                           batch_size = self.batchsize,
                                           shuffle =True)
        
        v = iter(loader)
        spectra, label = next(v)
        assert (list(spectra.shape) == [20, 1, 1, 101])
        assert (len(label)== 2)
        assert (len(label[0]) == self.batchsize)
        print('pass 4')
        
    def loader_test5(self):
        # test poly_c
        print('start 5')
        dataset = tano_signal.cube_loader(cube=self.cube, 
                                          fcnm=self.fcnm, 
                                          rhi=self.rhi, 
                                          x_transform=None, 
                                          y_transform=None, 
                                          PEV='poly_c')
    
        loader = torch.utils.data.DataLoader(dataset = dataset,
                                           batch_size = self.batchsize,
                                           shuffle =True)
        
        v = iter(loader)
        spectra, label = next(v)
        assert (list(spectra.shape)== [20, 1, 2, 101])
        assert (len(label)== 2)
        assert (len(label[0]) == self.batchsize)
        print('pass 5')
        
    def loader_test6(self):
        # test None 
        print('start 6')
        dataset = tano_signal.cube_loader(cube=self.cube, 
                                          fcnm=self.fcnm, 
                                          rhi=self.rhi, 
                                          x_transform=None, 
                                          y_transform=None, 
                                          PEV=None)
    
        loader = torch.utils.data.DataLoader(dataset = dataset,
                                           batch_size = self.batchsize,
                                           shuffle =True)
        
        v = iter(loader)
        spectra, label = next(v)
        assert (list(spectra.shape) == [20, 1, 1, 101])
        assert (len(label)== 2)
        assert (len(label[0]) == self.batchsize)
        print('pass 6')
        

"""
    model_predictor_test: A class to test model, weights and predictor.
    
    input Attributes
    ----------
    cube : array 
        cubic data
    fcnm: array
        Fcnm ground truth 
    rhi: array
        Rhi ground truth 


    Methods
    -------
    calculate_test1(): calculate the R-hi and F-cnm map using CNN with polynomial feature concatenation as 
                       representations and using pretrained poly_concate_c_weights. 
                       
    calculate_test2(): calculate the R-hi and F-cnm map using CNN-transformer with trainable PEV adding as 
                       representations and using pretrained learnable_PEV_ct_weights.
    
"""

        
        
class model_predictor_test():
    
    def __init__(self, cube, fcnm, rhi):
        self.cnn_weights = tano_signal.model.poly_concate_c_weights(device='cpu')
        self.ct_weights = tano_signal.model.learnable_PEV_ct_weights(device='cpu')
        self.fcnm = fcnm
        self.rhi = rhi
        self.cube = cube
        self.cnn = tano_signal.model.spectra_cnn(PEV= 'poly_c', 
                                                weights = self.cnn_weights,
                                                num_output=2, 
                                                drop_out_rate=0)
        
        self.cnn_transformer = tano_signal.model.spectra_cnn_transformer(PEV= 'trainable_a',
                                                                         weights = self.ct_weights,
                                                                         num_output=2, 
                                                                         drop_out_rate=0)
    
    def plott_(self, im, title, label):
        plt.figure(figsize=(10,10))
        plt.imshow(im, cmap='cividis')
        cbar = plt.colorbar(shrink=0.5, pad=0.005)
        cbar.set_label(f'${label}$', size=16)
        plt.title(f'{title}')
        plt.gca().invert_yaxis()
        plt.xlabel('X [Coordinate]')
        plt.ylabel('Y [Coordinate]')
        plt.show()
        
        
    def calculate_test1(self):
        # cnn
        print('test CNN')
        Fcnm, Rhi= calculate(self.cube, model = self.cnn, PEV= 'poly_c')
        self.plott_(im =self.fcnm, title = 'ground truth Fcnm', label='Fcnm')
        self.plott_(im = Fcnm, title = 'prediction Fcnm', label='Fcnm')
        self.plott_(im =self.rhi, title = 'ground truth rhi', label='rhi')
        self.plott_(im = Rhi, title = 'prediction rhi', label='rhi')
        
        
    def calculate_test2(self):
        # CNN-transformer 
        print('test CNN-transformer')
        Fcnm, Rhi= calculate(self.cube, model = self.cnn_transformer, PEV= 'trainable_a')
        self.plott_(im =self.fcnm, title = 'ground truth Fcnm', label='Fcnm')
        self.plott_(im = Fcnm, title = 'prediction Fcnm', label='Fcnm')
        self.plott_(im =self.rhi, title = 'ground truth rhi', label='rhi')
        self.plott_(im = Rhi, title = 'prediction rhi', label='rhi')