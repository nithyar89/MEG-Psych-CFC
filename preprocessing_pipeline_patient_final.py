# copyright Nik Murphy, PhD, Nithya Ramakrishnan, MS, CNTC, UTH, Houston 2017
#### MEG ASSR PRE-PROCESSING PIPELINE 
# import organisation functions
import os.path as op
# import mne functions
import mne
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.preprocessing.ica import ICA, run_ica
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.preprocessing import maxwell_filter
from mne.viz import plot_projs_topomap
from mne.io import RawArray
from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet
from mne.decoding import UnsupervisedSpatialFilter
from mne.chpi import _get_hpi_info, _calculate_chpi_positions
# import python scientific functions
import numpy as np
import scipy
from scipy import signal, stats
import copy
import random
from sklearn.decomposition import PCA, FastICA
# import visualisation functions
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# import meg processing functions
sys.path.append('/raid5/rcho/MEG_NM_NR_testing/FINALMNE/SCRIPTS/pre_processing/')
import meg_functions as mf

##### LOAD DATA AND PREPARE MONTAGE
#data_path = 'path with raw data files' #subject, raw, ssaep
#fname = 'example.fif' #ssaep.fif
data_path = '/raid5/rcho/PSYCH_CFC/MEG_raw_data/' #subject, raw, ssaep
#patient_ids = [1,2,3,4,5,6,7,8,9,13,17,18,19,20,21,22,23,24,28,29,30,32,34,35,36] #current ids in our workspace 
run =  5 #change this number as per the subjectid 
patient = "h%02d" % run
PATIENT = "H%02d" % run

fname = op.join(data_path, '%s_%s' %(patient, patient), 'ssaep/', '%s_SSAEP.fif' %(PATIENT)) #ssaep.fif

ica_path = '/raid5/rcho/MEG_NM_NR_testing/FINALMNE/DATA/ica/'
icaname =op.join(ica_path, '%s_rawica.fif' %(healthy)) #name of subj then rawica.fif
save_path='/raid5/rcho/MEG_NM_NR_testing/FINALMNE/DATA/epochs/'
saveall = op.join( '/raid5/rcho/MEG_NM_NR_testing/FINALMNE/DATA/epochs/%s_all-epo.fif' %(healthy))
save40 = op.join(save_path, '%s_40epo.fif' %(healthy))
save30 = op.join(save_path, '%s_30epo.fif' %(healthy))
save20 = op.join(save_path, '%s_20epo.fif' %(healthy))
raw = mf.load_MEG(data_path,fname)
##### GET CHPI 
qq = mf.get_chpi_info(raw)
##### ORGANISE EVENTS
events = mne.find_events(raw, stim_channel='STI101',shortest_event=1)
events = events[1:,:] # this deletes the first event as it is only 1 sample 
tmin = -1.5 # epoch baseline 
tmax = 1.5 # epoch end
baseline = (None, 0) # baseline correct from beginning to zero time
mne.viz.plot_events(events)
##### ASSR EVENTS
event_id = {'40hz':204,'30hz':194,'20hz':248} 
##### FILTER - MAX AND BANDPASS 
raw = mf.raw_filter(raw,1,1,qq,1,'in') 
###### FIND BAD CHANNELS
print(raw.info['bads'])
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True,
                       stim=False, exclude='bads')
reject = dict(mag=100e-12,grad=4000e-13) # reject extreme epochs
########## ICA ##########
method = 'extended-infomax'
eogch = 'EOG001' #eog001 or 2
ica = mf.ICA_decompose(raw,method=method,decim=3,variance=0.99,npcas=None,maxpcas=None,reject=reject,picks=picks)
mne.preprocessing.ICA.save(ica,icaname) 
ica = mf.ICA_artefacts(ica,raw,eogch=None,eog=0)
ica.plot_sources(raw) # plot projections of time series - continuous
manual_bad_comps = [# indices of bad components # ] # add indices
print(ica.exclude)
ica.exclude.extend(manual_bad_comps) # add in the manually selected components
print(ica.exclude)
ica.plot_overlay(raw)
ica.apply(raw)
##### Final Clean-up and Save ####
post_ica_epochs_all = mne.Epochs(raw,events,event_id,tmin,tmax,proj=True,picks=picks,
                   baseline=baseline,preload=True,
                   reject=reject,add_eeg_ref=False,reject_by_annotation=True)
post_ica_epochs_all.plot()
manual_bad_epochs = [5,6,12,13,18,25,26,77,105,128,132,165,202] #add indices
post_ica_epochs_all.drop(manual_bad_epochs,reason='user define',verbose=True)
mne.Epochs.save(post_ica_epochs_all,saveall)#!/usr/bin/env python2
#### NEW EPOCHING ROUTINE ########
post_ica_epochs_40 = post_ica_epochs_all['40hz']
mne.Epochs.save(post_ica_epochs_40,save40)#!/usr/bin/env python2
post_ica_epochs_30 = post_ica_epochs_all['30hz']
mne.Epochs.save(post_ica_epochs_30,save30)#!/usr/bin/env python2
post_ica_epochs_20 = post_ica_epochs_all['20hz']
mne.Epochs.save(post_ica_epochs_20,save20)#!/usr/bin/env python2
