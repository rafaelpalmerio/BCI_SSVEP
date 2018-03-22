# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 16:19:40 2017

@author: Rafael
"""
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import scipy.signal
#import kalman
import pywt
import numpy
import scipy
#import pywt&nbsp;
plotly.tools.set_credentials_file(username='RafaelPalmerio', api_key='pMHFCPMBYUmaK9PYhx3h')
import numpy as np
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api



def get_results(arquivo, amostra, freqs, debug=False):
    numero =0
    lista_index=[]
    df = pd.read_csv(arquivo)
    df = df.iloc[1000:-1000].reset_index()
    divisao_minima = 0.25
    tamanho_janela_segundos = 4
    
    tam_janela = int(tamanho_janela_segundos/divisao_minima)
    
    for index,row in df.iterrows():
        if numero>=250*divisao_minima:
            numero=1
        else:
            numero+=1
        lista_index.append(numero)
    df['index_num'] = lista_index
    #print(inicio, fim+1)
    
    # faz a media de valor lido por index
    #soma = df.groupby('index').mean()
    # define quais valores serão usados na fft
    #y = df[[canal,'index']].drop_duplicates('index', keep='last')[canal]
    #y = df[canal].iloc[2000:2256]
    #y = soma[canal]
    
    def media_movel(df, n=100):
        """
        Faz a média móvel do sinal
        """
        df['dummy'] = df.index//n
        df['media'] = df.groupby('dummy').mean()
        df = df.drop('media',axis=1).merge(df[['media']].dropna().reset_index(),left_on='dummy', right_on='index').\
                        drop(['dummy', 'index'],axis=1).rename(columns = {'final':'final_0'})
        df['final'] = df['final_0'] - df['media']
        return df[['final']]
    
    
    def pega_janelas(inicio, fim,df,canal):
        # divide o sinal em janelas do tempo especificado no inicio pelo seu valor de index
        numero=0
        lista_numero = []
        for index,row in df.iterrows():
            if row['index_num'] ==1:
                numero+=1
            lista_numero.append(numero)
        df['numero'] = lista_numero
        #print(inicio, fim+1)
        df = df[df['numero'].isin(range(inicio,fim+1))]
        soma = df.groupby('index_num').mean()
        y = soma[canal]
        return df,y
    
    def soma_canais(df,grupo_1, grupo_2, media=True):
        grupo_1 = list(map(lambda x: 'canal_' + str(x),grupo_1))
        grupo_2 = list(map(lambda x: 'canal_' + str(x),grupo_2))
        df = df[grupo_1+grupo_2]
        if media:
            df = df - df.mean()
        df_grupo_1 = df[grupo_1].mean(axis=1)
        df_grupo_2 = df[grupo_2].mean(axis=1)
        return df_grupo_1-df_grupo_2
        
        
    Wn = [4/125,20/125]
    Wn =0.16
    #aplica o filtro de butterworth
    def butterworth(y, Wn=20/125,N=5, tipo = 'high'):
        b, a = scipy.signal.butter(N, Wn, tipo)
        y = scipy.signal.filtfilt(b, a, y)
        return y
    
    #Fs = 255.0;  # sampling rate
    
    def fft(Fs,y):
        Ts = 1.0/Fs; # sampling interval
        
        #ff = 5;   # frequency of the signal
        #y = np.sin(2*np.pi*ff*t)
        #aplica a fft no sinal filtrado
        t = np.arange(0,Ts*len(y),Ts) # time vector
        
        
        n = len(y) # length of the signal
        if n % 2 ==1:
            n=n-1
        #print(n)
        k = np.arange(n)
        #print(k)
        T = n/Fs
        frq = k/T # two sides frequency range
        frq = frq[range(n//2)] # one side frequency range
        
        Y = np.fft.fft(y)/n # fft computing and normalization
        Y = Y[range(n//2)]
        return frq,abs(Y),t
    
    
    def hanning(y):
        han = np.hanning(y.shape[0])
        return han
    
    def blackman(y):
        return np.blackman(y.shape[0])
    
    def hamming(y):
        ham = np.hamming(y.shape[0])
        return ham
    
    def flattop(y, sym=True):
        ft = scipy.signal.flattop(y.shape[0], sym=sym)
        return ft
    
    def subtrai_media(y):
        return y-y.mean()
    
    def wl(y,name):
        cA, cD = pywt.dwt(y, name)
        #y2 = pywt.idwt(cA, cD, name)
        return cA,cD
    
    def wavelet(y,name):
        noiseSigma = 0.16
        levels  = int( np.floor( np.log2(y.shape[0]) ) )-3
        coeffs = pywt.wavedec(data=y, wavelet=name, level=levels)
        threshold = noiseSigma*np.sqrt(2*np.log2(y.size))
        NewWaveletCoeffs = list(map (lambda x: pywt.threshold(x,threshold, mode='soft'),
                                coeffs))
        #print(NewWaveletCoeffs)
        #print(levels)
        New_y = pywt.waverec( NewWaveletCoeffs, wavelet=name)
        return New_y
    
    def get_output_array(amostra, tipo = 'numerico'):
        # fazo vetor saída baseado na key do dicionario
        if tipo == 'vetorial':
            output = np.array([[0,0,0,0,0]])
            if '7' in amostra:
                output[0][0] = 1
            elif '9' in amostra:
                output[0][1] = 1
            elif '11' in amostra:
                output[0][2] = 1
            elif '13' in amostra:
                output[0][3] = 1
            elif '15' in amostra:
                output[0][4] = 1        
        elif tipo == 'numerico':
            if '7' in amostra:
                output = np.array([0])
            elif '9' in amostra:
                output = np.array([1])
            elif '11' in amostra:
                output = np.array([2])
            elif '13' in amostra:
                output = np.array([3])
            elif '15' in amostra:
                output = np.array([4])
        elif tipo == 'resposta':
            output = int(amostra.split('_')[0])
        return output
    
    def filtra_apply(row):
        for freq in freqs:
            if row['frequencia_olhos']<freq+1 and row['frequencia_olhos']>freq-1:
                return freq
        return None

    
    #define o canal de leitura
    canal='canal_1'
    
    
    janelas = df[df.index_num==1].shape[0]
    
    # df é o df original, z é só o canal especificado
    df, z=pega_janelas(1,janelas,df,canal)
    
    
    #divide os canais em grupos pela proximidade física
    grupo_1 = [1,2,3,4,5]
    grupo_2 = [6]
    
    #soma os canais pelos grupos especificados acima
    df['final'] = soma_canais(df, grupo_1, grupo_2, media= True)
    
    #inicializando os containers de dados
    results_matrix = np.matlib.zeros((freqs.__len__(), freqs.__len__()))
    
    #pegando um canal só
    #df['final'] = df[canal]
    for i in range(1,janelas+1):
        #df, y=pega_janelas(i,5,df,canal)
        #y=soma[canal]
        #y=df[canal]
        
        #y = df[canal][df['numero']==i].reset_index(drop=True)
        y = df[['final']][df['numero'].isin(range(i, i+tam_janela))]\
                        .reset_index(drop=True)
        #y = df[['final']][df]
        
        if y.__len__() < 250*tamanho_janela_segundos:
            continue
        
        
        #y=media_movel(y)
        #y = subtrai_media(y)       
    
        y = y['final']
        #y=butterworth(y,Wn=0.8, tipo='low')
        #y=butterworth(y,Wn=0.07, tipo='high', N=8)
        
        y = y*flattop(y)
        
        y = wavelet(y,'coif1')
    
    
        #y=butterworth(y,Wn=0.4)
        
        #y = kalman.kalman(y,n_iter=len(y))
        
        
        #y = y*flat_top(y, False)
        #t = np.arange(0,100, 0.01)
        #y = np.sin(t)
        
        frq,Y,t=fft(250,y)
        #plota a fft do sinal filtrado
        #fig, ax = plt.subplots(2, 1)
        #ax[0].plot(t,y)
        #ax[0].set_xlabel('Time')
        #ax[0].set_ylabel('Amplitude')
        
        #t = np.arange(0,100, 0.01)
    
#        plt.plot(t,y)
#        plt.xlabel('Time')
#        plt.ylabel('Amplitude')
#        plt.show()
        #ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
        #ax[1].set_xlabel('Freq (Hz)')
        #ax[1].set_ylabel('|Y(freq)|')
        
        #plot_url = py.plot_mpl(fig, filename='mpl-basic-fft')
        
        # pega as frequencias de maior amplitude na fft do sinal filtrado
        kappa = pd.DataFrame()
        kappa['frequencia_olhos'] = frq
        kappa['amplitude'] = Y
        
        kappa = kappa.iloc[0*tamanho_janela_segundos:20*tamanho_janela_segundos]
        output_desejado = int(get_output_array(amostra, tipo='resposta'))
        
        kappa['frequencia'] = kappa.apply(filtra_apply, axis=1)

        kappa = kappa[~kappa['frequencia'].isnull()].reset_index(drop=True)
        
        maximo = kappa[kappa.amplitude==kappa.amplitude.max()].iloc[0]
        output_real = int(maximo['frequencia'])
        
        row_index = freqs.index(output_desejado)
        col_index = freqs.index(output_real)
        #print(maximo)
        #print('Máximo: ',fr_max, 'Resposta: ', output )
        results_matrix[row_index, col_index] += 1
        if debug:
            print(kappa[(kappa['frequencia_olhos']<17) & (kappa['frequencia_olhos']>6)]\
              .sort_values('amplitude',ascending=False).reset_index(drop=True).iloc[0:40])
    #return dataset, final_output
    return results_matrix


filenames = {'8_0':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-teste_8hz.csv',
             '12_0':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-teste_12hz.csv',
             '10':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-teste_olhos_fechados.csv',
             '8_1':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20170918_8hz.csv',
             '15_0':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20170918_15hz.csv',
             '7':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20170918_7hz.csv',
             '7_1': r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20170921_7hz.csv',
             '9':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20170921_9hz.csv',
             '11':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20170921_11hz.csv',
             '13':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20170921_13hz.csv',
             '15_1':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20170921_15hz.csv',
             '7_2':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20171005_7hz.csv',
             '7_3':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20171005_7hz_2.csv',
             '9_2':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20171005_9hz.csv',
             '9_3':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20171005_9hz_2.csv',
             '11_2':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20171005_11hz.csv',
             '11_3':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20171005_11hz_2.csv',
             '13_2':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20171005_13hz.csv',
             '13_3':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20171005_13hz_2.csv',
             '15_2':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20171005_15hz.csv',
             '15_3':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20171005_15hz_2.csv'
                 }


usaveis = ['15_0', '7', '7_1', '9', '11', '13', '15_1']
usaveis = [ '7_2', '7_3', '9_2', '9_3', '11_2', '11_3','13_2', '13_3', '15_2', '15_3']

usaveis = [ '7','7_1','7_2', '7_3', '9', '9_2', '9_3','11', '11_2', '11_3',
           '13','13_2','13_3','15_0', '15_2', '15_3']

#usaveis = [ '7_2', '7_3','11_2', '11_3',\
#           '13_2', '13_3', '15_2', '15_3']

#usaveis = [ '7_2', '7_3', '9_2', '9_3', \
#            '13_2', '13_3', '15_2', '15_3']

freqs = list(set(list(map(lambda x: int(x.split('_')[0]), usaveis))))
results_matrix = np.matlib.zeros((freqs.__len__(), freqs.__len__()))
for index, amostra in enumerate(usaveis):
    print(amostra)
    results_matrix = results_matrix + get_results(filenames[amostra], amostra, freqs)
    """
    if index == 0:
        final_dataset = kap
        final_output = output
    else:
        final_dataset = (np.concatenate((final_dataset,
                                         kap), axis=0))
        final_output = (np.concatenate((final_output,
                                         output), axis=0))
    """
        
def recall_precision(matrix):
    size = matrix.shape[0]
    acertos = 0
    df = pd.DataFrame(matrix)
    score_df = pd.DataFrame(columns=['precision', 'recall', 'support'])
    for line in range(size):
        acertos += df[line].iloc[line]
        row = {}
        row['precision'] = df[line].iloc[line]/df[line].sum()
        row['recall'] = df[line].iloc[line]/df.iloc[line].sum()
        row['support'] = int(df.iloc[line].sum())
        score_df = score_df.append(row, ignore_index=True)
    score_df['support'] = score_df['support'].astype(int)    
    avg_total = score_df.mean()
    avg_total['support'] = int(score_df['support'].sum())
    avg_total['overall_precision'] = acertos/avg_total['support']
    return score_df, avg_total
        
confusion_matrix = recall_precision(results_matrix)

print(results_matrix)
print()
print(confusion_matrix[0])
print()
print(confusion_matrix[1])