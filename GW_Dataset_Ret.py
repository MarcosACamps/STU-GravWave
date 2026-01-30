# -*- coding: utf-8 -*-
"""
=Known Issues =
- May stop runnning without ending run.
- Code stopped partway through 'H1' after successfully
    going through 'L1'
    (possible error from looping too many times?)


@author: Marcos
"""
#%% MODULES
import matplotlib.pyplot as plt
import os
import csv

import gwosc
from gwosc.locate import get_urls
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
import requests


# Specify the directory path where you want to save the file
directory = '--directory--'

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(directory):
    os.makedirs(directory)
#%% EVENT CHECK

#print(gwosc.datasets.find_datasets(type='events', detector='L1'))

#%% BBH Events

#events = gwosc.datasets.find_datasets(type='events', detector='L1')

detector_list = ['L1', 'H1']


#%% Create Q
for i in range(len(detector_list)):
    detector = detector_list[i]
    events = gwosc.datasets.find_datasets(type='events', detector=detector)
    for x in range(len(events)):
        event = events[x]
        
        gps = event_gps(event)
        start = int(gps) -10
        end = int(gps) +10
        
        url = get_urls(detector, gps, gps)[-1]
        
        fn = os.path.basename(url)
        
        with open(fn, 'wb') as strainfile:
            straindata = requests.get(url)
            strainfile.write(straindata.content)
            
        strain = TimeSeries.read(fn, format='hdf5.gwosc')
        center = int(gps)
        strain = strain.crop(center-16,center+16)
        
        file_name = detector+'_'+event + '_sample_' + str(i) + '.png'
        
        dt = 1
        strain = strain.whiten(4,2)
        hq = strain.q_transform(frange = (30,500), qrange=(20,50), outseg=(gps-dt, gps+dt))
        fig = hq.plot()
        ax = fig.gca()
        #fig.colorbar(label="Normalized Energy")
        ax.grid(False)
        #ax.set_yscale('log')
        ax.axis('off')
        plt.savefig(directory+'/'+file_name, bbox_inches='tight')
        
        


        print(f"File '{file_name}' saved to '{directory}'.")

