#!/usr/bin/env python

# contains universal functions
# [requirement]
# python packages: pytorch numpy pandas xarray
# bash lib: nco

import os 
import torch
import numpy as np
import pandas as pd
import xarray as xr

# this is an nc writing function
# if the nc file already exist, add or overwrite according to the time given; if not, using nco to generate nc file according to the cdl file given, then add or overwritte
def write_nc(nc_file, data_list, var_list, dims_list, timestr, cdl_file=None):
    
    # transfer data into numpy.array form when necessary
    def check_numpy(data):
        if isinstance(data, torch.Tensor):
            data.detach().numpy()
        return data
    
    # write data seperately
    def write_var(ds, data, var_name, dims):
        data = check_numpy(data)
        data_array = xr.DataArray(data, dims=dims, name=var_name)
        if var_name==dims: # write dimensions
            ds = ds.assign_coords({var_name:(var_name,data_array.values.astype(np.float32),ds[var_name].attrs)})
        else: # write variables
            data_array['time'] = new_time_da
            da_back = ds[var_name]
            for tm in new_time:
                da_back[da_back.time == tm] = data_array.sel(time=tm)
            ds[var_name].values = da_back.values.astype(np.float32)
            ds[var_name].encoding['_FillValue'] = None
        return ds


    # generate nc from cdl
    if cdl_file!=None:
        os.system('ncgen -o '+nc_file+' '+cdl_file) 
        print('Created '+nc_file+'!')
    print('Ready to write variables into '+nc_file+'!')
    
    ds = xr.open_dataset(nc_file,engine='netcdf4')
    # deal with time dimension
    if isinstance(timestr, list):
        new_time = np.squeeze([pd.to_datetime(timestr)])
    else:
        new_time = [pd.to_datetime(timestr)]
    new_time_da = xr.DataArray(new_time, dims='time', name='time', coords=[new_time])
    uniq_time = np.unique([item for item in ds.time.values if np.count_nonzero(item == new_time)==0])
    full_time_da = xr.concat([ds.time.sel(time=uniq_time), new_time_da], dim='time')
    ds = ds.reindex(time=full_time_da).sortby('time')
    
    # write data
    if isinstance(data_list, list):
        for data, var_name, dims in zip(data_list, var_list, dims_list):
            ds = write_var(ds, data, var_name, dims)
            print('Wrote : '+var_name+'!')
    else:
        ds = write_var(ds, data_list, var_list, dims_list)
        print('Wrote : '+var_list+'!')
        
    # transfer into nc file
    os.system('rm -rf '+nc_file)
    ds.encoding['_FillValue'] = None
    ds.to_netcdf(nc_file)
    ds.close()



# this is a function extract data from an nc file
# only capable to select time piece
def getdata_nc(nc_file, var_list, timestr=None, lon_sel=None, lat_sel=None,lonname_list=['lon', 'longitude'],latname_list=['lat', 'latitude']):
    # get data from nc
    data = []
    ds = xr.open_dataset(nc_file,engine='netcdf4')
    lon_name = ''.join([lonname for lonname in lonname_list if lonname in ds.keys()])
    lat_name = ''.join([latname for latname in latname_list if latname in ds.keys()])
    if isinstance(var_list, list)==0:
        var_list = [var_list]
    if isinstance(timestr, type(None))==0:
        times_num = np.datetime64(timestr)
        ds = ds.sel(time=times_num)
    for var in var_list:
        da = ds[var]
        if isinstance(lon_sel, type(None))==0:
            da = eval("da.sel("+"".join(lon_name)+"=lon_sel, method='nearest')")
        if isinstance(lat_sel, type(None))==0:
            da = eval("da.sel("+"".join(lat_name)+"=lat_sel, method='nearest')")
        data.append(np.squeeze(da.values))
    ds.close()
    return data
    


# this is a function extract longitude and latitude  from an nc file
def getll_nc(nc_file,lonname_list=['lon', 'longitude'],latname_list=['lat', 'latitude']):
    # get longitude and latitude from nc
    ds = xr.open_dataset(nc_file,engine='netcdf4')
    lon_name = ''.join([lonname for lonname in lonname_list if lonname in ds.keys()])
    lat_name = ''.join([latname for latname in latname_list if latname in ds.keys()])
    lon, lat = ds[lon_name].values, ds[lat_name].values
    ds.close()
    return lon,lat
