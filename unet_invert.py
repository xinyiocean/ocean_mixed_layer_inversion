#!/usr/bin/env python
import torch
import numpy as np
from datetime import datetime, timedelta
from classes import net,normalizer
from tools.pre_function import write_nc,getll_nc,getdata_nc


### settings
input_size, output_size = 6, 1
filters_num = [8,16,32,64,128]
modelfile = 'data_file/UNET_models/unet_expt7_06.pth' 
file_dict = {'invs_fn':'data_file/Input_Case4_1/inver_set/HYCOM_DATE.npz', 
             'mean_nc':'data_file/HYCOM_mean/HYCOMnwp_mld_DATE_MD_avg_daily.nc',
             'mask_nc':'_mask/HYCOMnwp_MLD_mask.nc',
             'cdl_fn': 'cdl/HYCOMnwp_mld.cdl',
             'rslt_nc':'HYCOM_inv_DATE.nc'} 
var_dict = {'features': ['sst','ssh','u10','v10'],
            'labels': 'mld',
            'mean_var': 'MLD_Tdiff',
            'mask_var': 'LANDMASK',
            'rslt_var': ['longitude','latitude','MLD_Tdiff_est','MLD_Tdiff_dia','MLDa_Tdiff_est','MLDa_Tdiff_dia'],
            'rslt_dim': ['longitude','latitude',('time','latitude','lontitude'),('time','latitude','lontitude'),\
                        ('time','latitude','lontitude'),('time','latitude','lontitude')]}
start_date = datetime(2019, 2, 1)
end_date = datetime(2019, 2, 15)
cut_inx1,cut_inx2 = 11,12

### load lon, lat and mask
lon,lat = getll_nc(file_dict['mask_nc'])
londata,latdata = torch.meshgrid(torch.tensor(lon,dtype=torch.float32),\
                                 torch.tensor(lat,dtype=torch.float32),\
                                 indexing='xy')
ldmask = getdata_nc(file_dict['mask_nc'], 'LANDMASK')[0]
ldmask = torch.tensor(ldmask,dtype=torch.int)==1
londata, latdata = londata.masked_fill(ldmask, torch.nan), latdata.masked_fill(ldmask, torch.nan)
ldmask = ldmask.detach().numpy()

### creat normalization instance
feature_nmer = normalizer.ZScoreNorm(input_size)
label_nmer = normalizer.ZScoreNorm(output_size)

feature_nmer.mean_value = [8.7084915e+01, 7.7194991e+00, -3.4361130e-07, 2.3884544e-07, 1.6971501e-07, 1.7988327e-06] # changeable
feature_nmer.std_value = [65.73861, 16.400911, 0.57587314, 0.08399373, 2.9168887, 2.703061] # changeable
label_nmer.mean_value = [-1.11286345e-05] # changeable
label_nmer.std_value = [11.951486] # changeable


### creat net model
model = net.Unet(input_size, output_size, filters_num)
load_dict = torch.load(modelfile, map_location=torch.device('cpu'))
model.load_state_dict(load_dict)
model.eval()

current_date = start_date
while current_date <= end_date:
    ### dateinfo
    datestr = current_date.strftime('%Y%m%d')
    current_date += timedelta(days=1)
    datestr_MD = datestr[4:]
    datestr_Y = datestr[:4]
    
    ### load mean data
    mean_fn = file_dict['mean_nc'].replace('DATE_MD',datestr_MD)
    mndata = getdata_nc(mean_fn, var_dict['mean_var'], lon_sel=lon, lat_sel=lat)[0]

    ### invert
    invs_fn = file_dict['invs_fn'].replace('DATE',datestr)
    ds_invs = np.load(invs_fn)
    # load features
    ftdata = []
    for var in var_dict['features']:
        ftdata.append(torch.tensor(ds_invs[var], dtype=torch.float32, requires_grad=False))
    ftdata = torch.stack(ftdata,0)
    # load labels
    lbdata = torch.tensor(ds_invs[var_dict['labels']], dtype=torch.float32, requires_grad=False)
    # unsqueeze
    ftdata = ftdata.unsqueeze(0)
    lbdata = lbdata.unsqueeze(0).unsqueeze(0)
    # add lon and lat
    lldata = torch.stack([londata,latdata],dim=0).unsqueeze(0)
    ftdata = torch.cat([lldata,ftdata],dim=1)
    # resize
    ftdata = ftdata[:,:,cut_inx1:,cut_inx2:]
    lbdata = lbdata[:,:,cut_inx1:,cut_inx2:]
    # fill nan
    ftdata[torch.isnan(ftdata)] = 0
    # invert
    ftdata = feature_nmer.Norm(ftdata)
    ivdata = model(ftdata)
    ivdata = label_nmer.Denorm(ivdata)
    ivdata[torch.isnan(lbdata)] = torch.nan
    
    ### process all data
    ivdata = ivdata.squeeze().detach().numpy()
    lbdata = lbdata.squeeze().detach().numpy()
    ivdata = np.pad(ivdata, ((cut_inx1,0),(cut_inx2,0)), "constant")
    lbdata = np.pad(lbdata, ((cut_inx1,0),(cut_inx2,0)), "constant")
    ivdata[ldmask], lbdata[ldmask] = np.nan, np.nan
    ivdata, lbdata, mndata = np.expand_dims(ivdata,axis=0), np.expand_dims(lbdata,axis=0), np.expand_dims(mndata,axis=0)
    # remove extreme
    iv_mld = mndata+ivdata
    lb_mld = mndata+lbdata
    iv_mld[iv_mld<0]=0
    lb_mld[lb_mld<0]=0
    ivdata,lbdata = iv_mld-mndata, lb_mld-mndata
    
    
    ### write result
    rslt_fn = file_dict['rslt_nc'].replace('DATE',datestr)
    write_nc(nc_file=rslt_fn, data_list=[lon,lat,mndata+ivdata,mndata+lbdata,ivdata,lbdata],\
             var_list=var_dict['rslt_var'], dims_list=var_dict['rslt_dim'], \
             timestr=datestr, cdl_file=file_dict['cdl_fn'])
    
