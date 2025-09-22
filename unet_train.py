#!/usr/bin/env python

import os
import torch
import torch.nn as nn
import time
from classes import net,normalizer,dataset
from tools.torch_function import bp_train,test
from tools.pre_function import getll_nc,getdata_nc
from torch.utils.data import DataLoader

### GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
device_ids = [0,1]

### settings
input_size, output_size = 6, 1
filters_num = [8,16,32,64,128]  #filters_num should be list type
epoch_size, batch_size, learnrate, target = 200, 24, 0.001, 0.01
path_train = 'Input_Case4_1/train_set/'
path_test = 'Input_Case4_1/test_set/'
modelfile = 'unet_case4_1_example.pth'
size1,size2 = 11,12

### prepare lon and lat
llfile = '_mask/HYCOMnwp_mld_mask.nc' # changeable
lon,lat = getll_nc(llfile)
lon,lat = torch.meshgrid(torch.tensor(lon,dtype=torch.float32),\
                         torch.tensor(lat,dtype=torch.float32),\
                         indexing='xy')
ldmask = getdata_nc(llfile, 'LANDMASK')[0]
ldmask = torch.tensor(ldmask,dtype=torch.int)==1
lon, lat = lon.masked_fill(ldmask, torch.nan), lat.masked_fill(ldmask, torch.nan)

### creat net model
model = net.Unet(input_size, output_size, filters_num)
loss = nn.MSELoss(reduction='mean')
optim = torch.optim.Adam(params=model.parameters(), lr=learnrate)

### creat normalization instance
feature_nmer = normalizer.ZScoreNorm(input_size)
label_nmer = normalizer.ZScoreNorm(output_size)

### creat dataloader instance
dataloader_train = DataLoader(dataset = dataset.dataset_npz(path_train, ['sst','ssh','u10','v10'],'mld'),
                              batch_size = batch_size,
                              shuffle = True,
                              drop_last = True)
dataloader_test = DataLoader(dataset = dataset.dataset_npz(path_test, ['sst','ssh','u10','v10'],'mld'),
                             batch_size = batch_size,
                             shuffle = False,
                             drop_last = False)


### transport net to GPU
model = nn.DataParallel(model, device_ids=device_ids) # changeable
model.to(device)
torch.cuda.synchronize()

# print information
print('Unet experiment7 2016-2018,2020-2022')
print('Features: lon, lat, SST, SSH, U10m, V10m \nLabel: MLD \nMean: daily')
print('Number of features: %d\nNumber of labels: %d\nNumber of conv layers: %d' %  (input_size,output_size,len(filters_num)))
print('Number of filters:', filters_num)
print('Other settings: \nbatch size=%d, learning rate=%7.5f, target MSEloss=%6.4f, loss=MSE_mean, optim=Adam' % (batch_size,learnrate,target))

### training
print('\n>>>Training starts!!!')
for epoch in range(epoch_size):
    start = time.time()
    ls_train, ls_test, batch_inx = 0, 0, 0
    trainsam_count, testsam_count = 0, 0
    print('\n<Epoch: %d START>' % (epoch+1))
    # train
    for ftdata, lbdata in dataloader_train:
        batch_inx += 1
        batch_count = ftdata.shape[0]
        # append lon and lat
        lldata = torch.stack([lon,lat],dim=0).unsqueeze(0)
        lldata = lldata.repeat(batch_count,1,1,1)
        ftdata = torch.cat([lldata,ftdata],dim=1)
        # resize
        ftdata = ftdata[:,:,size1:,size2:]
        lbdata = lbdata[:,:,size1:,size2:] 
        # fill nan
        ftdata[torch.isnan(ftdata)] = 0
        lbdata[torch.isnan(lbdata)] = 0
        # remove extreme
        ftdata[:,:,torch.squeeze(torch.any(torch.abs(lbdata)>500,dim=0))] = 0
        lbdata[:,:,torch.squeeze(torch.any(torch.abs(lbdata)>500,dim=0))] = 0
        # normalize
        ftdata = feature_nmer.BatchNorm(ftdata)
        lbdata = label_nmer.BatchNorm(lbdata)
        ftdata, lbdata = ftdata.to(device), lbdata.to(device)
        # train
        model.train()
        output, ls, model = bp_train(ftdata, lbdata, model, loss, optim)
        ls_train += ls*batch_count
        trainsam_count += batch_count
        print('batch number %d, loss: %7.4f' % (batch_inx,ls))
        torch.cuda.synchronize()
    # test
    for ftdata, lbdata in dataloader_test:
        batch_count = ftdata.shape[0]
        # append lon and lat
        lldata = torch.stack([lon,lat],dim=0).unsqueeze(0)
        lldata = lldata.repeat(batch_count,1,1,1)
        ftdata = torch.cat([lldata,ftdata],dim=1)
        # resize
        ftdata = ftdata[:,:,size1:,size2:] 
        lbdata = lbdata[:,:,size1:,size2:]
        # fill nan
        ftdata[torch.isnan(ftdata)] = 0
        lbdata[torch.isnan(lbdata)] = 0
        # remove extreme
        ftdata[:,:,torch.squeeze(torch.any(torch.abs(lbdata)>500,dim=0))] = 0
        lbdata[:,:,torch.squeeze(torch.any(torch.abs(lbdata)>500,dim=0))] = 0
        # normalize
        ftdata = feature_nmer.Norm(ftdata)
        lbdata = label_nmer.Norm(lbdata)
        ftdata, lbdata = ftdata.to(device), lbdata.to(device)
        # test
        model.eval()
        output, ls = test(ftdata, lbdata, model, loss)
        ls_test += ls*batch_count
        testsam_count += batch_count
        torch.cuda.synchronize()
    print('Training loss: %7.4f\nTesting loss: %7.4f\nCost time: %ds'
           % (ls_train/trainsam_count, ls_test/testsam_count, time.time()-start))
    # saving net
    torch.save(model.module.state_dict(), modelfile)
    if ls_train <= target:
        print('Acheive target loss!!!')
        break
    print('\nNormalization Information:')
    print('[Features]')
    feature_nmer.info()
    print('[Label]')
    label_nmer.info()
print('\nTraining finished!!!<<<')
print('< Save as '+modelfile+ ' >')

