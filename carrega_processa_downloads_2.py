# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 22:37:44 2017

@author: Rafael
"""

import os
import glob
import pandas as pd
import re

os.chdir(r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\download_eeg_data_2')

arquivos = glob.glob(os.path.join('multi','*EEG*'))
d = {''}
labels_ = []

for arquivo in arquivos:    
    labels_ = labels_ + open(arquivo.replace('EEG', 'Target'), 'r').read().strip().split(',')
    
  
un = list(set(labels_))
un.sort()
freq_dict = {key:str(i+1) for i, key in enumerate(un)}


last_user = ''
for arquivo in arquivos:
    user = re.findall('Sub(.*?)_', arquivo)[0]
    filename_base = os.path.join(os.getcwd(), 'data',str(int(user)).zfill(3),'h{}hz.csv')
    if last_user != user:
        freq_count = {key:0 for key in freq_dict.values()}
    data = pd.read_csv(arquivo, header=None)
    
    labels = open(arquivo.replace('EEG', 'Target'), 'r').read().strip().split(',')
    
    header = list(map(lambda x: str(freq_dict[x]), labels))
    data.columns = [str(i) for i in range(data.shape[1])]

    for col, freq in zip(data.columns, header):
        temp = data[[col]].copy()
        name = str(freq) + '_' + str(freq_count[freq])
        filename = filename_base.format(name)
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        temp.rename(columns = {col:'canal_1'}).to_csv(filename, index=False)
        freq_count[freq] +=1