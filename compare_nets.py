# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:36:16 2017

@author: Rafael
"""
import pickle
import json

filenames = [r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\2017-10-31_10h52\rede_neural2.pkl',
             r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\2017-10-31_10h52\rede_neural.pkl']

for filename in filenames:
    with open(filename, 'rb') as f:
        rede_ = pickle.load(f)
        with open(filename.replace('pkl', 'txt'), 'w') as f2:
            f2.write(str(rede_.coefs_))
            
with open(filename.replace('.pkl', 'original.txt'), 'w') as f2:
    f2.write(str(rede.coefs_))