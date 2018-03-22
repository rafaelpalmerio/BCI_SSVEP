# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 16:19:40 2017

@author: Rafael
"""
import pandas as pd


def get_results(arquivo):
   
    numero =0
    lista_index=[]
    df = pd.read_csv(arquivo)
    tamanho_janela_segundos = 4
    
    for index,row in df.iterrows():
        if numero>=250*tamanho_janela_segundos:
            numero=1
        else:
            numero+=1
        lista_index.append(numero)
    df['index_num'] = lista_index
    #print(inicio, fim+1)
    
    # faz a media de valor lido por index
    #soma = df.groupby('index').mean()
    
    
    import matplotlib.pyplot as plt
    import plotly
    import scipy.signal
    import kalman
    import pywt
    import numpy
    import scipy
    #import pywt&nbsp;
    plotly.tools.set_credentials_file(username='RafaelPalmerio', api_key='pMHFCPMBYUmaK9PYhx3h')
    import numpy as np
    # Learn about API authentication here: https://plot.ly/python/getting-started
    # Find your api_key here: https://plot.ly/settings/api




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
    
    def soma_canais(df,grupo_1, grupo_2):
        grupo_1 = list(map(lambda x: 'canal_' + str(x),grupo_1))
        grupo_2 = list(map(lambda x: 'canal_' + str(x),grupo_2))
    
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
        print(levels)
        coeffs = pywt.wavedec(data=y, wavelet=name, level=levels)
        threshold = noiseSigma*np.sqrt(2*np.log2(y.size))
        NewWaveletCoeffs = list(map (lambda x: pywt.threshold(x,threshold, mode='soft'),
                                coeffs))
        #print(NewWaveletCoeffs)
        #print(levels)
        New_y = pywt.waverec( NewWaveletCoeffs, wavelet=name)
        return New_y
    
    
    
    canal='canal_4'
    janelas = df[df.index_num==1].shape[0]
    df, z=pega_janelas(1,janelas,df,canal)
    
    grupo_1 = [1,2,3,4,5]
    grupo_2 = [6]
    
    df['final'] = soma_canais(df, grupo_1, grupo_2)
    #df['final'] = df[canal]
    for i in range(1,janelas+1):
        #define o canal de leitura
        #canal='canal_2'
        #df, y=pega_janelas(i,5,df,canal)
        #y=soma[canal]
        #y=df[canal]
        
        #y = df[canal][df['numero']==i].reset_index(drop=True)
        
        y = df[['final']][df['numero']==i].reset_index(drop=True)
        
        if y.__len__() < 250*tamanho_janela_segundos:
            continue
        
        #y=media_movel(y)
        y = subtrai_media(y)
    
        
    
        
        y=butterworth(y['final'],Wn=0.1, tipo='low')
        y=butterworth(y,Wn=0.1, tipo='high')
        
        #y = y*flattop(y)
        
        #y = wavelet(y,'coif1')
    
    
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
    
        plt.plot(t,y)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()
        #ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
        #ax[1].set_xlabel('Freq (Hz)')
        #ax[1].set_ylabel('|Y(freq)|')
        
        #plot_url = py.plot_mpl(fig, filename='mpl-basic-fft')
        
        # pega as frequencias de maior amplitude na fft do sinal filtrado
        kappa = pd.DataFrame()
        kappa['frequencia_olhos'] = frq
        kappa['amplitude'] = Y
        
        #print(kappa.iloc[0:25])
        print(kappa[(kappa['frequencia_olhos']<17) & (kappa['frequencia_olhos']>6)]\
              .sort_values('amplitude',ascending=False).reset_index(drop=True).iloc[0:40])
        return kappa


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
             '15_1':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-20170921_15hz.csv'
                 }

#le o arquivo e coloca em um dataframe
usaveis = ['15_0', '7', '7_1', '9', '11', '13', '15_1']
kap=get_results(filenames['15_0'])    
#for amostra in usaveis

