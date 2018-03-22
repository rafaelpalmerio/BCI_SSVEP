# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 19:19:06 2017

@author: Rafael
"""
import glob
import re
import processa_dados_eeg_neural_network_real_time
import neural_network

folder = 'C:\\Users\\Rafael\\Documents\\TCC\\BCI\\EEG_Data\\2017-10-22_20h56'
files = glob.glob(folder+r'\*.csv')
file_dic = {re.findall('h[0-9]{2}_(.*?)hz',file)[0]:file for file in files }

dataset, output = processa_dados_eeg_neural_network_real_time.roda_processamento(filenames_dic = file_dic)#, source = 'jk')
rede = neural_network.rede_neural(dataset,output)