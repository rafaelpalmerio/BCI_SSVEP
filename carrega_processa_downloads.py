# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:33:47 2017

@author: Rafael
"""

import scipy.io
import glob
import re
import pandas as pd
import os
import numpy as np

def traduz(estado):    
    if len(estado) == 3:
        key = ['4', '3', '2', '4', '1', '2', '5', '3', '4', '1', '3', '1', '3']
    elif len(estado) == 2:
        key = ['4', '2', '3', '5', '1', '2', '5', '4', '2', '3', '1', '5']
    else:
        key = ['5', '3', '2', '1', '4', '5', '2', '1', '4', '3']    
    
    return key

def pega_frequencias(df, inicios, fins, key):
    df['freq'] = np.nan
    for i in enumerate(zip(inicios, fins)):
        ind = i[0]
        freq = key[ind]
        inicio = i[1][0]
        fim = i[1][1]
        df['freq'].iloc[inicio:fim] = freq
    df['freq'] = df['freq'].fillna('6')
    return df

def salva_arquivo(df, events, usuario, freq_dict):
    #######CUIDADO AQUI######
    events = events[events['event_id'].isin([32779,32780])]
    #########################
    # processando df para separar nas frequencias
    df_proc = (df.reset_index().rename(columns={'index':'line'}).
            merge(events.drop(['count', 'count2'], axis=1), on='line', how='left').drop('line', axis=1))
    df_proc['event_id'] = df_proc['event_id'].fillna(0)
    df_proc['aq'] = df_proc['event_id'].cumsum()
    df_proc = df_proc.drop('event_id', axis=1)
    
    # iterando nas diferentes partes da aquisicao
    for aq in df_proc['aq'].unique():
        if aq==0: continue
        temp = df_proc[df_proc['aq']==aq].copy()
        
        current_freq = str(int(temp['freq'].unique()[0]))
        if current_freq == '6': continue
        name = current_freq + '_' + str(freq_dict[current_freq])
        
        # salvando o arquivo
        filename = os.path.join(os.getcwd(),'download_eeg_data',usuario, 'h' + name + 'hz.csv')
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        temp.drop('aq', axis=1).to_csv(filename, index=False)
        freq_dict[current_freq] +=1 
    
    return freq_dict

def main():
    last_usuario = 0
    arquivos = glob.glob(os.path.join(os.getcwd(), 'download_eeg_data','EEG-SSVEP-Experiment3', '*.mat'))
    for arquivo in arquivos:
        # pega o usuario
        usuario = str(re.findall('U([0-9]{3})\D', arquivo)[0])
        if last_usuario != usuario:
            freq_dict = {str(i):0 for i in range(1,7)}
        last_usuario = usuario
        
        # pega a frequencia
        estado = str(re.findall('U[0-9]{3}(.*?)\.', arquivo)[0])
        
        # pegando a key
        key = traduz(estado)
        
        # carregando o arquivo .mat (dicionario)
        mat = scipy.io.loadmat(arquivo)
        
        # carregando em um dataframe e renomeando as colunas
        df = pd.DataFrame(mat['eeg'].transpose())
        df.columns = ['canal_' + str(i+1) for i, _ in enumerate(df.columns)]
        
        # pegando a tabela de eventos
        events = pd.DataFrame(mat['events'], columns=['count', 'event_id','line', 'count2'])
        events['line'] = events['line']-1
        
        # pegando os eventos de inicio e fim para preencher as frequencias
        inicios = list(events['line'][events['event_id']==32779])
        fins = list(events['line'][events['event_id']==32780])
        
        # preenchendo a tabela de eeg com as frequencias
        df = pega_frequencias(df.copy(), inicios, fins, key)
        
        freq_dict = salva_arquivo(df, events, usuario, freq_dict)
        
    

