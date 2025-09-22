#!/usr/bin/env python

# contains dataset class in pytorch Dataset form
# the output data is in torch.tensor type; default stacking dimension is 0
# [requirement]:
# python packages: pytorch


import os
import torch
import numpy as np
from torch.utils.data import Dataset

# the input data should be in numpy npz form
class dataset_npz(Dataset):
    
    def __init__(self, path, featurename, labelname, prefix='', postfix='npz', stack_dim=0):
        fnlist = os.listdir(path)
        fn = [fn for fn in fnlist if fn.startswith(prefix) and fn.endswith(postfix)]
        fn.sort()
        self.filelist = fn
        self.path = path
        if isinstance(featurename,list)==0:
            featurename = [featurename]
        if isinstance(labelname,list)==0:
            labelname = [labelname]
        self.ftname = featurename
        self.lbname = labelname
        self.stack_dim = stack_dim

    def __getitem__(self, index):
        # open single npz file
        filename, path = self.filelist[index], self.path
        dataSet = np.load(path+filename)
        # read in features
        fdata_list = []
        for ft in self.ftname:
            # read in feature
            fdata = torch.tensor(dataSet[ft], dtype=torch.float32)
            # append features
            fdata_list.append(fdata)
        featureData = torch.stack(fdata_list,axis=self.stack_dim)
        # read in labels
        ldata_list = []
        for lb in self.lbname:
            # read in label
            ldata = torch.tensor(dataSet[lb], dtype=torch.float32, requires_grad=False)
            # append labels
            ldata_list.append(ldata)
        labelData = torch.stack(ldata_list,axis=self.stack_dim)
        return featureData, labelData

    def __len__(self):
        return len(self.filelist)
