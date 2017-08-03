# CFC Pipeline - sensor level

# Import and setup environment
import sys
sys.path.append('/raid5/rcho/MEG_NM_NR_testing/time_frequency_modules/')
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.io
import scipy.signal
from scipy.signal import butter, filtfilt, hilbert
from numpy import random
import mne
import cmath
import copy
import tf_module1 as tf1
import tf_module2 as tf2
import pickle
import tfclass_creation as tfc
# organise files for loading
class Admin(object):
    datapath='path'
    savepath='path'
    files = []
Admin.datapath = '/raid5/rcho/MEG_NM_NR_testing/FINALMNE/DATA/epochs/'
Admin.savepath = '/raid5/rcho/MEG_NM_NR_testing/FINALMNE/DATA/sensor_level_timef/'
Admin.files = os.listdir(Admin.datapath)
# prune files list and add paths
files = [x for x in Admin.files if "all" in x]
Admin.files = files
for x in range(len(Admin.files)):
    Admin.files[x] = Admin.datapath+Admin.files[x] # add full path to names
del files
class Epochs(object):
    def __init__(self,datapath,hz40,hz30,hz20,chans):
        self.datapath = datapath
        self.e40hz = hz40
        self.e30hz = hz30
        self.e20hz = hz20
        self.chans = chans
def make_Epochs(datapath,hz40,hz30,hz20,chans):
    return Epochs(datapath,hz40,hz30,hz20,chans)
class BinData(object):
    def __init__(self,datapath,TG40,TG30,TG20,DG40,DG30,DG20):
        self.datapath = datapath
        self.TG40 = TG40
        self.TG30 = TG30
        self.TG20 = TG20
        self.DG40 = DG40
        self.DG30 = DG30
        self.DG20 = DG20
def make_BinData(datapath,TG40,TG30,TG20,DG40,DG30,DG20):
    return BinData(datapath,TG40,TG30,TG20,DG40,DG30,DG20)
class WavData(object):
    def __init__(self,datapath,wav40,wav30,wav20):
        self.wav40 = wav40
        self.wav30 = wav30
        self.wav20 = wav20
def make_WavData(datapath,wav40,wav30,wav20):
    return WavData(datapath,wav40,wav30,wav20)


# this will now loop through all files (irrespective of condition) and apply processing
for subj in range(1,len(Admin.files)):
    print('currently running item ' + str(subj))
    name = Admin.files[subj][51:58]
    savename = Admin.savepath+name
    epochs =mne.read_epochs(Admin.files[subj], proj=True, preload=True, verbose=None) # correction to epoch rejection 5/10/17 - nik
    evs = epochs.event_id   
    ev = [x for x in evs if "hz" in x]  
    evs = [u'20hz',u'30hz',u'40hz']  
    events = []
    for x in range(0,len(ev)):
        e = ev[x]
        for y in range(0,len(evs)):
            match = e==evs[y]
            if match==True:
                events.append(e)
    missing=[]
    for x in range(0,len(events)):
        a = evs[x] in events
        if a==False:
            missing.append(evs[x])
        chans = []
    # create epochs and convert to tesla
    for x in range(0,len(events)):
        t = str(events[x])
        epochs_temp = epochs[t]
        nom = 'epochs'+'_'+t[0:2]
        tt=nom+' = tf1.fem_to_tesla(epochs_temp._data)'
        exec(tt)
        if nom == 'epochs_40':
            chans = tf1.chan_select(np.mean(epochs_40,axis = 0),np.arange(50,1301,1),np.arange(1750,2501,1),95,1000,40,5)
#    if len(chans)==0:
#        print "40hz condition does not exist"
#        continue # 40hz condition does not exist, move on to the next subject
    ## EPOCHING ##
    # create everything else
    for x in range(0,len(events)):
        t = str(events[x])
        nom = 'epochs_'+t[0:2]
        tt = nom+'=np.squeeze(np.mean('+nom+'[:,chans,:],axis = 1))'
        exec(tt)
    # create missing data
    if len(missing)>0:
        for x in range(0,len(missing)):
            m = str(missing[x])
            nom = 'epochs_'+m[0:2]
            tt = nom+'="does not exist"'
            exec(tt)
        del m
    epdata = make_Epochs(Admin.files[subj],epochs_40,epochs_30,epochs_20,chans)
    del chans, epochs_20,epochs_30,epochs_40,nom,tt,t 
    ## WAVELETS ##
    for x in range(0,len(events)):
        t = str(events[x])
        nom = 'epdata.e'+t[0:4]
        nom2 = 'wav'+t[0:2]
        tt = nom2+'=tf2.erp_wavelets('+nom+',(2,82),2,5,(500,1000))'
        exec(tt)
    if len(missing)>0:
        for x in range(0,len(missing)):
            m = str(missing[x])
            nom = 'epdata.e'+m[0:4]
            nom2 = 'wav'+m[0:2]
            tt = nom2+'="does not exist"'
            exec(tt) 
        del m
    # save timef data
    waveletdata = make_WavData(Admin.files[subj],wav40,wav30,wav20) 
    np.save(savename+'_timef',waveletdata,allow_pickle = True)
    del waveletdata,wav40,wav30,wav20,nom,nom2,tt,t
    ## CFC ## - stimulation period
    for x in range(0,len(events)):
        t = str(events[x])
        nom = 'epdata.e'+t[0:4]
        nom2 = 'khdataTG'+t[0:2]
        nom3 = 'khdataDG'+t[0:2]
        tt = nom2+'=tf2.kldiv_spec_band('+nom+',60,200,False,(4,8),(32,48),1000,(1501,2601))'
        tt2 = nom3+'=tf2.kldiv_spec_band('+nom+',60,200,False,(4,8),(32,48),1000,(1501,2601))'
        exec(tt)
        exec(tt2)
    if len(missing)>0:
        for x in range(0,len(missing)):
            m = str(missing[x])
            nom = 'epdata.e'+m[0:4]
            nom2 = 'khdataTG'+m[0:2]
            nom3 = 'khdataDG'+m[0:2]
            tt = nom2+'="does not exist"'
            tt2 = nom3+'="does not exist"'
            exec(tt)
            exec(tt2)
        del m
    stim_sensor_CFC = make_BinData(Admin.files[subj],khdataTG40,khdataTG30,khdataTG20,khdataDG40,khdataDG30,khdataDG20)
    np.save(savename+'_cfcStim',stim_sensor_CFC,allow_pickle = True)
    del khdataTG40,khdataTG30,khdataTG20,khdataDG40,khdataDG30,khdataDG20,tt,t,nom,nom2,nom3   
    ## CFC ## - baseline period
    for x in range(0,len(events)):
        t = str(events[x])
        nom = 'epdata.e'+t[0:4]
        nom2 = 'khdataTG'+t[0:2]
        nom3 = 'khdataDG'+t[0:2]
        tt = nom2+'=tf2.kldiv_spec_band('+nom+',60,200,False,(4,8),(32,48),1000,(500,1500))'
        tt2 = nom3+'=tf2.kldiv_spec_band('+nom+',60,200,False,(4,8),(32,48),1000,(500,1500))'
        exec(tt)
        exec(tt2)
    if len(missing)>0:
        for x in range(0,len(missing)):
            m = str(missing[x])
            nom = 'epdata.e'+m[0:4]
            nom2 = 'khdataTG'+m[0:2]
            nom3 = 'khdataDG'+m[0:2]
            tt = nom2+'="does not exist"'
            tt2 = nom3+'="does not exist"'
            exec(tt)
            exec(tt2)
        del m
    bl_sensor_CFC = make_BinData(Admin.files[subj],khdataTG40,khdataTG30,khdataTG20,khdataDG40,khdataDG30,khdataDG20)
    np.save(savename+'_cfcBaseline',bl_sensor_CFC,allow_pickle = True)
    del khdataTG40,khdataTG30,khdataTG20,khdataDG40,khdataDG30,khdataDG20,stim_sensor_CFC,tt,t,nom,nom2,nom3      
    
            








