# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 16:19:40 2017

@author: Rafael and Lucas
"""
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import scipy.signal
import kalman
import neural_network
import pywt
import numpy
import scipy
#import pywt&nbsp;
#plotly.tools.set_credentials_file(username='RafaelPalmerio', api_key='pMHFCPMBYUmaK9PYhx3h')
import numpy as np

##############################
#       Processamento        #
##############################

def media_movel(df, n=128, n_canais=8):
    """
    Faz a média móvel do sinal
    """
    # colunas dos valores
    canais = ['canal_' + str(i) for i in range(1,n_canais+1)]
    colunas_medias = ['media_' + str(i) for i in range(1,n_canais+1)]
    
    # fazendo a coluna dummy para agrupamento das linhas
    temp = df.copy()
    temp['dummy'] = temp.index//n
    
    # agrupando pelo dummy  e renomeando as colunas
    media = temp.groupby('dummy').mean().rename(columns = {a:b for a,b in zip(canais, colunas_medias)}).reset_index()
    media = media[['dummy'] + colunas_medias]
    
    # left join para pegar os valores de media
    temp = temp.merge(media, on='dummy', how='left')
    
    # tirando o valor da media de cada canal em cada janela
    for canal, media in zip(canais, colunas_medias):
        temp[canal] -= temp[media]
    
    # dropando as colunas auxiliares
    temp = temp.drop(colunas_medias + ['dummy'], axis=1)
    
    return temp
    
def pega_janelas(inicio, fim,df,canal):
    """
    divide o sinal em janelas do tempo especificado no inicio pelo seu valor de index
    """
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
    
def soma_canais(df,grupo_1, grupo_2, media=False):
    """
    soma os grupos de canais pela sua localização na cabeça
    """
    grupo_1 = ['canal_' + str(x) for x in grupo_1]
    grupo_2 = ['canal_' + str(x) for x in grupo_2]
    df = df[grupo_1+grupo_2]
    
    if media:
        df = df - df.mean()
    df_grupo_1 = df[grupo_1].sum(axis=1)
    df_grupo_2 = df[grupo_2].sum(axis=1)
    res = (df_grupo_1**2-df_grupo_2)

    return res

def tira_pontas(df, s_comeco, s_fim):
    """
    Funcção que tira s_comeco segundos do comeco do dataframe e s_fim do final
    """
    lista = list(df.index[df['index_num']==1])
    index_inicio = lista[s_comeco]
    index_fim = lista[-s_fim]
    df = df.iloc[index_inicio:index_fim]
    return df
        
def normalize_channels(df, n_canais=8):
    cols = ['canal_'+str(i) for i in range(1,n_canais+1)]
    for col in cols:
        if df[col].std() != 0:
            df[col]=(df[col]-df[col].mean())/df[col].std()
    return df
    
def fft(Fs,y):
    """ 
    faz a fft do sinal
    """
    # tamanho do sinal
    n = len(y)
    if n % 2 == 1:
        n=n-1
        # ajustando o tamanho do vetor para coincidirem os tamanhos
        y = y[:-1]
    
    Ts = 1.0/Fs; # sampling interval
    
    #aplica a fft no sinal filtrado
    t = np.arange(0,Ts*len(y),Ts) # time vector
    
    #print(n)
    k = np.arange(n)
    #print(k)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range
    
    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(n//2)]
    return frq,abs(Y),t
    
def butterworth(y, Wn=20/125,N=5, tipo = 'high'):
    """
    aplica o filtro de butterworth
    """
    b, a = scipy.signal.butter(N, Wn, tipo)
    y = scipy.signal.filtfilt(b, a, y)
    return y

def filtra_sinal(y, high_Wn = 0.1, high_N = 8,
                       low_Wn=0.1, low_N = 8, hann=False, hamm=False, fltop=False, wvlt=False):
    """
    função que aplica os filtros de maneira uniforme
    """    
    if low_Wn:          
        y=butterworth(y,Wn=low_Wn, tipo='low', N=low_N)
    if high_Wn:
        y=butterworth(y,Wn=high_Wn, tipo='high', N=high_N)
    
    if hann:
        y = y*hanning(y)
    
    if hamm:
        y = y*hamming(y)

    if fltop:
        y = y*flattop(y)
        
    if wvlt:
        y = wavelet(y,'coif1')    
    #y = kalman.kalman(y,n_iter=len(y))  
    
    return y

def filtra_sinal_todo(df, high_Wn = 0.1, high_N = 8,
                       low_Wn=0.1, low_N = 8, hann=False, hamm=False, fltop=False, wvlt=False, n_canais=8):
    """
    função que aplica os filtros de maneira uniforme
    """
    cols = ['canal_'+str(i) for i in range(1,9)]
    for col in cols:
        
        if low_Wn:          
            df[col]=butterworth(df[col],Wn=low_Wn, tipo='low', N=low_N)
        if high_Wn:
            df[col]=butterworth(df[col],Wn=high_Wn, tipo='high', N=high_N)
        
        if hann:
            df[col] = df[col]*hanning(df[col])
        
        if hamm:
            df[col] = df[col]*hamming(df[col])
    
        if fltop:
            df[col] = df[col]*flattop(df[col])
            
        if wvlt:
            df[col] = wavelet(df[col],'coif1')    
    #y = kalman.kalman(y,n_iter=len(y))  
    
    return df

def get_output_array(amostra, tipo = 'numerico', source = 'tempo_real'):
    """
    faz o vetor saída baseado na key do dicionario
    """
    if source != 'download':
        if tipo == 'vetorial':
            output = np.array([[0,0,0,0,0]])
            if '100':
                output[0][5] = 1
            elif '7' in amostra:
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
            if '100' in amostra:
                output = np.array([5])
            elif '7' in amostra:
                output = np.array([0])
            elif '9' in amostra:
                output = np.array([1])
            elif '11' in amostra:
                output = np.array([2])
            elif '13' in amostra:
                output = np.array([3])
            elif '15' in amostra:
                output = np.array([4])
    else:
        output = np.array([int(amostra)])
    return output

def plot_fft(frq,Y,t,y):
    #plota a fft do sinal filtrado
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t,y)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')

    plt.plot(t,y)
    
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    plt.plot(frq,Y)
    ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.show()
    
    #plot_url = py.plot_mpl(fig, filename='mpl-basic-fft')

##########################
#       Windowing        #
##########################

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

######################################################################
#       Processa uma amostra vinda de aquisição em tempo real        #
######################################################################

def processa_real_time(df, tamanho_janela_segundos=4, high_Wn = 0.1, high_N = 8,
                       low_Wn=0.1, low_N = 8, hann=False, hamm=False, fltop=False, wvlt=False):
    #media movel
    df = media_movel(df)
    
    # divide o sinal em dois grupos
    grupo_1 = [1,2,3,4,5]
    grupo_2 = [6]
    
    #soma os canais pelos grupos especificados acima
    df['final'] = soma_canais(df, grupo_1, grupo_2, media= True)
    
    #y=soma[canal]
    #y=df[canal]
    
    #y = df[['final']][df]
    
    #y = subtrai_media(y)       

    y = df['final']
    
    y = filtra_sinal(y, high_Wn = high_Wn, high_N = high_N,
                       low_Wn=low_Wn, low_N = low_N, hann=hann, hamm=hamm, fltop=fltop, wvlt=wvlt)
    
    
    #y = y*flat_top(y, False)
    frq,Y,t=fft(float(y.shape[0])/tamanho_janela_segundos,y)
    #frq,Y,t=fft(float(250,y)
    #plot_fft()
    
    # pega as frequencias de maior amplitude na fft do sinal filtrado
    kappa = pd.DataFrame()
    kappa['frequencia_olhos'] = frq
    kappa['amplitude'] = Y
    
    kappa = kappa.iloc[0*tamanho_janela_segundos:25*tamanho_janela_segundos]
    
    dataset = np.array(kappa[['amplitude']]).transpose()
        
    #print(kappa[(kappa['frequencia_olhos']<17) & (kappa['frequencia_olhos']>6)]\
    #      .sort_values('amplitude',ascending=False).reset_index(drop=True).iloc[0:40])
    return dataset
    
    

################################################
#       Processa uma amostra de arquivos       #
################################################

def processa_from_file(arquivo, amostra, divisao_minima = 1, 
                       tamanho_janela_segundos = 4, source = 'tempo_real',
                       s_comeco = 2, s_fim = 2, 
                       high_Wn = 0.1, high_N = 8,
                       low_Wn=0.1, low_N = 8, hann=False, hamm=True, fltop=False, wvlt=False):
    # fft separada: fft é feita por canal e seus vetores são juntados
    fft_separada = True
    
    # junto: o processamento do sinal é feito no sinal todo, e não nas janelas
    junto=False
    if junto and fft_separada:
        assert False
    aux_bool = True   
    freq_aq = 250
    n_canais = 8
    
    numero =0
    lista_index=[]
    df = pd.read_csv(arquivo)
    
    
    # para cada caso, as frequencias podem ser diferentes
    if source == 'download':
        if df.shape[1] > 2:
            freq_aq = 128
            n_canais=14
        else:
            freq_aq = 512
            n_canais=1
    
    #df = normalize_channels(df, n_canais=n_canais)
    df = media_movel(df, n_canais=n_canais)
    
    
    if not source == 'tempo_real':
        for index,row in df.iterrows():
            if numero>=freq_aq*divisao_minima:
                numero=1
            else:
                numero+=1
            lista_index.append(numero)
        df['index_num'] = lista_index
    
    
    if source != 'download':
        df = tira_pontas(df, s_comeco, s_fim)
    else:
        #df = tira_pontas(df, 1, 1)    
        pass
    tam_janela = int(tamanho_janela_segundos/divisao_minima)
    
    #print(inicio, fim+1)
    
    # faz a media de valor lido por index
    #soma = df.groupby('index').mean()
    
    #y = soma[canal]  
        
    #define o canal de leitura
    canal='canal_1'
    
    
    janelas = df[df.index_num==1].shape[0]
    # df é o df original, z é só o canal especificado
    df, z=pega_janelas(1,janelas,df,canal)    
    
    #divide os canais em grupos pela proximidade física
    if source == 'download' and n_canais == 14:
        grupo_1 = [5,6,7,8,9,10]
        grupo_2 = [1,2,3,4,11,12,13,14]
    else:
        grupo_1 = [1,2,3,4,5]
        grupo_2 = [6] 
    #soma os canais pelos grupos especificados acima
    if not junto and not fft_separada:
        df['final'] = soma_canais(df, grupo_1, grupo_2, media= False)

        
    
    #frq,Y,t=fft(float(df['final'].shape[0])/tamanho_janela_segundos,df['final'])
    #plot_fft(frq,Y,t,df['final'])
    
    #inicializando os containers de dados
    dataset = []
    final_output = []
    
    #pegando um canal só
    #df['final'] = df[canal]
    for i in range(1,janelas+1):
        #y=soma[canal]
        #y=df[canal]
        #y = df[canal][df['numero']==i].reset_index(drop=True)
        y = df[df['numero'].isin(range(i, i+tam_janela))]\
                        .reset_index(drop=True).copy()

        if junto:            
            y = filtra_sinal_todo(y, high_Wn = high_Wn, high_N = high_N,
                           low_Wn=low_Wn, low_N = low_N, hann=hann, hamm=hamm, fltop=fltop, wvlt=wvlt, n_canais=n_canais)
            if n_canais > 1:
                y['final'] = soma_canais(y, grupo_1, grupo_2, media= False)
        
        if y.__len__() < freq_aq*tamanho_janela_segundos and source != 'tempo_real':
            continue
        elif y.__len__() < 0.9*freq_aq*tamanho_janela_segundos and source == 'tempo_real':
            continue
        
        if fft_separada:
            df_new = pd.DataFrame()
            canais = ['canal_'+str(i) for i in range(1, n_canais+1)]
            for ch in canais:
                # caso um canal esteja ruim no download
                if y[ch].nunique() < 10 and source == 'download':                    
                    aux_bool= False
                    continue
                this = filtra_sinal(y[ch], high_Wn = high_Wn, high_N = high_N,
                       low_Wn=low_Wn, low_N = low_N, hann=hann, hamm=hamm, fltop=fltop, wvlt=wvlt)
                
                frq, Y, t = fft(float(this.shape[0])/tamanho_janela_segundos,this)
                df_new[ch] = Y
            if aux_bool and n_canais != 1:
                df_new['final'] = soma_canais(df_new, grupo_1, grupo_2, media= False)
            elif aux_bool:
                print(df_new.columns)
                df_new['final'] = df_new['canal_1']
            df_new['frq'] = frq
            y = df_new.copy()
            #print(amostra)           
            #plot_fft(frq,y['final'],frq,y['final'])
            
        if not aux_bool:
            aux_bool=True
            continue
        #y = subtrai_media(y)
        y = y['final']
        
        if not junto and not fft_separada:
            y = filtra_sinal(y, high_Wn = high_Wn, high_N = high_N,
                       low_Wn=low_Wn, low_N = low_N, hann=hann, hamm=hamm, fltop=fltop, wvlt=wvlt)
        
        kappa = pd.DataFrame()
        if not fft_separada:
            frq,Y,t=fft(float(y.shape[0])/tamanho_janela_segundos,y)
            kappa['frequencia_olhos'] = frq
            kappa['amplitude'] = Y
        else:
            kappa['frequencia_olhos'] = df_new['frq']
            kappa['amplitude'] = df_new['final']
        #print(amostra, kappa.sort_values('amplitude',ascending=False).head())
        #plot_fft()
        
        # pega as frequencias de maior amplitude na fft do sinal filtrado      
        kappa = kappa.iloc[0*tamanho_janela_segundos:28*tamanho_janela_segundos]
        #print(kappa.shape)
        output = get_output_array(amostra, source=source)

        if type(dataset) == numpy.ndarray:
            dataset = np.concatenate((dataset,np.array(kappa[['amplitude']]).transpose()), axis=0)
            final_output = np.concatenate((final_output,output), axis=0)
        else:
            dataset = np.array(kappa[['amplitude']]).transpose()
            final_output = output
            
        #print(kappa[(kappa['frequencia_olhos']<17) & (kappa['frequencia_olhos']>6)]\
        #      .sort_values('amplitude',ascending=False).reset_index(drop=True).iloc[0:40])
        #print(kappa.sort_values('amplitude', ascending=False))
        
    return dataset, final_output



def roda_processamento(filenames_dic = [], usaveis = [],divisao_minima = 1, 
                       tamanho_janela_segundos = 4, source = 'tempo_real',
                       high_Wn = 0.1, high_N = 8, low_Wn=0.1, 
                       low_N = 8, hann=False, hamm=False, fltop=False, wvlt=False):
    if filenames_dic == []:
        filenames_dic = {'8_0':r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\OpenBCI-RAW-teste_8hz.csv',
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
    elif usaveis ==[]:
        usaveis = filenames_dic.keys()
        
        #usaveis = ['9_0', '9_1', '11_0', '11_1', '13_0', '13_1', '100_0','100_1']
        #usaveis = [ '7_2', '7_3','11_2', '11_3',\
        #           '13_2', '13_3', '15_2', '15_3']
        
        #usaveis = [ '7_2', '7_3', '9_2', '9_3', \
        #            '13_2', '13_3', '15_2', '15_3']
    for index, am in enumerate(usaveis):
        amostra = am.split('_')[0]
        #print(amostra)
        kap, output=processa_from_file(filenames_dic[am], amostra, divisao_minima, tamanho_janela_segundos,
                                       source=source, high_Wn = high_Wn, 
                                       high_N = high_N,
                                       low_Wn=low_Wn, low_N = low_N, 
                                       hann=hann, hamm=hamm, fltop=fltop, wvlt=wvlt)
        if index == 0:
            final_dataset = kap
            final_output = output
        else:
            final_dataset = (np.concatenate((final_dataset,
                                             kap), axis=0))
            final_output = (np.concatenate((final_output,
                                             output), axis=0))
    return final_dataset, final_output

if __name__=="__main__":
    pass
    size = 100
    hidden_layers = 5
    hl_tuple = tuple(size for _ in range(hidden_layers))
    #final_dataset, final_output = roda_processamento(source = 'not_tempo_real')
    #rede = neural_network.rede_neural(final_dataset, final_output, hl_tuple)

        
        


