#!/usr/bin/env python

# contains normalizer classes
# the input data should be a torch.tensor type
# [requirement]:
# python packages: pytorch


import torch
# z-score normalizer
class ZScoreNorm:
    
    def __init__(self, size, mean_dim=[0,2,3]):
        self.name = 'Z-Score Normalization'
        self.size = size
        self.mean_value = torch.zeros(self.size)
        self.std_value = torch.zeros(self.size)
        self.batch_count = 0
        self.mean_dim = mean_dim
        
    def BatchNorm(self, data):
        self.renew(data)
        self.norm(data)
        return data

    def Norm(self, data):
        self.norm(data)
        return data

    def Denorm(self, data):
        self.denorm(data)
        return data
    
    def renew(self, data):
        if self.batch_count == 0:
            self.mean_value = torch.mean(data, dim=self.mean_dim)
            self.std_value = torch.std(data, dim=self.mean_dim, unbiased=False)
        else:
            mean_tmp = torch.mean(data, dim=self.mean_dim)
            std_tmp = torch.std(data, dim=self.mean_dim, unbiased=False)
            for num in range(self.size):
                if abs(mean_tmp[num])<=10*abs(self.mean_value[num]) and abs(std_tmp[num])<=10*abs(self.std_value[num]):
                    self.mean_value[num] = 0.9*self.mean_value[num]+0.1*mean_tmp[num]
                    self.std_value[num] = 0.9*self.std_value[num]+0.1*std_tmp[num]
        self.batch_count += 1

    def norm(self, data):
        for num in range(self.size):
            meanval, stdval = self.mean_value[num], self.std_value[num]
            data[:,num,:,:] = (data[:,num,:,:]-meanval)/stdval
    
    def denorm(self, data):
        for num in range(self.size):
            meanval, stdval = self.mean_value[num], self.std_value[num]
            data[:,num,:,:] = data[:,num,:,:]*stdval+meanval
                
    def info(self):
        print('Normalization Method: ', self.name)
        print('Mean Value: ', (self.mean_value.detach().numpy()))
        print('Std Value: ', (self.std_value.detach().numpy()))

