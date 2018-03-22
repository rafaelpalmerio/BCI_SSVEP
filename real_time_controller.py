# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:24:32 2017

@author: Rafael and Lucas
"""

import os
from datetime import datetime
import processa_dados_eeg_neural_network_real_time as processa
import neural_network
import svm_classifier
import play_audio
import csv
import sys; sys.path.append('..')  # make sure that pylsl is found (note: in a normal program you would bundle pylsl with the program)
import pylsl
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import inspect
import random
import glob
import re
import pandas as pd
import numpy as np
import pickle
import pca
from grid_search import grid_search

def tempo():
    return float((datetime.utcnow() - datetime(1970, 1, 1)).total_seconds())

def setup():
    # criando a pasta e o nome base dos arquivos
    now = datetime.now()
    tm = str(now).split()[0] +'_'+ str(now.hour).zfill(2)+'h'+str(now.minute).zfill(2)
    folder = os.path.join(os.getcwd(),'aquisicao_' + tm)
    path_base = os.path.join(folder,tm+'_{}hz.csv')
    if not os.path.isdir(folder):
        os.makedirs(folder)
    
    # numero de aquisicoes para cada frequencia 
    n_aq = 1
    
    # duracao de cada aquisicao
    duracao = 8
    frequencias = [7,9] # 100 é repouso
    file_dic = ({key:None for key in
                [i for j in list(map(lambda x: 
                    [str(x)+'_'+str(i) for i in range(n_aq)], frequencias)) 
                for i in j]})
    cabecalho = 'index_num,canal_1,canal_2,canal_3,canal_4,canal_5,canal_6,canal_7,canal_8,timestamp'
    return path_base, n_aq, duracao, file_dic, cabecalho.split(',')

def abortable_conn(func, *args, **kwargs):
    # função que tentar rodar a connect, e para depois de timeout
    
    # pegando o timeout da partial que a chama
    timeout = kwargs.get('timeout', None)
    
    # iniciando a thread que vai rodar a connect
    p = ThreadPool(1)
    
    # apply_async nessa thread
    res = p.apply_async(func, args=args)
    try:
        # espera timeout segundo para a connect ser completada
        out = res.get(timeout)
        return out
    except multiprocessing.TimeoutError:
        # caso dê timeout, levanta um erro
        print("Não foi possível estabeler conexão com o OpenBCI!")
        p.terminate()
        raise

def connect_controller():
    # função que controla a chamada da abortable_conn, que por sua vez chama a connect
    # quando for conectar ao openbci, chamar ESTA FUNÇÃO
    try:
        # instancia abortable_conn(func = connect, kwargs = timeout)
        abortable_func = partial(abortable_conn, connect, timeout=10)
        
        #rodando a função
        conn = abortable_func()
        return conn
    except:
        return 
    
def connect():
    # função que efetivamente conecta com o open BCI
    if inspect.stack()[1][3] != 'worker':
        print('ATENÇÃO! Não chamar o connect diretamente! Usar o connect_controller')
        return
    
    #  first resolve an EEG stream on the lab network
    print("Procurando pela stream EEG...")
    streams = pylsl.resolve_stream('type','EEG')
    
    # create a new inlet to read from the stream
    inlet = pylsl.stream_inlet(streams[0])
    
    #print(timestamp, list(sample))
    while True:
        tm = inlet.pull_sample([])
        if tm[0]:
            break
    return streams
    
def training_aquisition(streams, divisao_minima = 1):
    # pegando os parâmetros de teste
    path_base, naq, duracao, file_dic, cabecalho = setup()
    #inlet, sample = connect_controller()
    
    end = False
    # preenche todos os arquivos de treinamento
    while not end:
        # pega a lista das aquisições que ainda não foram
        available = list(filter(lambda x: file_dic[x] is None, file_dic.keys()))
        
        # se todas tiverem sido preenchidas, sai
        if not available:
            end=True
            break
        
        # sorteia uma frequencia
        current = random.choice(available)
        filename = path_base.format(current)
        
        # abre o arquivo de saida
        f = open(filename, 'w')
        csv_writer = csv.writer(f, lineterminator = '\n')
        csv_writer.writerow(cabecalho)
        
        # avisa o usuario para olhar para uma determinada luz
        print('\n\n\nOlhe para o led de %s Hertz.' %(current.split('_')[0]))
        play_audio.play_file(current.split('_')[0])
        
        # definindo o inlet
        inlet = pylsl.stream_inlet(streams[0])
        
        # aquisitando
        
        start = tempo()
        last_it = tempo()
        index = 0
        while tempo() - start <=  duracao:
            sample = []
            it_time = tempo()
            if it_time - last_it >= divisao_minima:
                last_it = tempo()
                index = 0
            timestamp = inlet.pull_sample(sample)
            if timestamp[0]:
                index += 1
                csv_writer.writerow([index]+timestamp[0]+[timestamp[1]])
        
        print('Pode parar de olhar.')
        
        # fechando o arquivo
        f.close()
        file_dic[current] = filename
    
    play_audio.play_file('Fim')
    return file_dic, path_base

def get_file_dic(folder = '', keyword='aquisicao', dataset = 0 ):
    if not folder:
        folders = glob.glob(os.path.join(os.getcwd(), keyword + '*'))
        folders.sort()
        folder = folders[-1]
    print('Usando os arquivos da pasta ', folder)
    """2017-11-05_17h35 ficou muito bom"""
    files = glob.glob(folder+r'\*hz.csv')
    if keyword:
        file_dic = {re.findall('h[0-9]{2}_(.*?)hz',file)[0]:file for file in files}
    else:
        file_dic = {re.findall('h(.*?)hz',file)[0]:file for file in files}
    return file_dic

def run_training(streams, method, args, divisao_minima = 1, tamanho_janela_segundos = 4, 
                 high_Wn = 0.1, high_N = 8, low_Wn=0.1, 
                 low_N = 8, hann=False, hamm=False, fltop=False, wvlt=False):
    
    # chamando a aquisição dos arquivos
    file_dic, path_base = training_aquisition(streams)
    
    # processando os sinais
    dataset, output = processa.roda_processamento(
            filenames_dic = file_dic,
            divisao_minima = divisao_minima,
            tamanho_janela_segundos = tamanho_janela_segundos,
            high_Wn = high_Wn, high_N = high_N,
            low_Wn=low_Wn,low_N = low_N, hann=hann, hamm=hamm, fltop=fltop, wvlt=wvlt)
    
    args = tuple((dataset, output)) + args
    rede, scaler, result = method(*args)
    print(file_dic)
    return rede, scaler, path_base, file_dic

def passa_lista(lista, element, size):
    if len(lista) == size:
        lista_return = [lista[i+1] for i in range(0, size-1)] + [element]
    else:
        lista_return = lista + [element]
    return lista_return

def run_debug_mode(rede, scaler, module, streams, path,
                           divisao_minima = 1, tamanho_janela_segundos = 4,
                           size = 10, full=False,
                           high_Wn = 0.1, high_N = 8, low_Wn=0.1, 
                           low_N = 8, hann=False, hamm=False, fltop=False, wvlt=False):
    possible_freqs = {'7':0, '9':1, '11':2, '13':3, '100':5}
    i = 0
    
    duration = 30
    n = 1
    aq=0
    
    print(path)
    
    # vetor para guardar o sinal do open bci
    signal = []
    
    # index da posição de cada sinal na janela
    index = 0
    
    # lista que guardará os indexes maximos das janelas anteriores
    index_lista = []
    
    # quantos outputs são guardados
    output_size = 100
    # lista para guardar esses outputs
    output_lista = []
    
    # cabecalho do dataframe
    cabecalho = 'index_num,canal_1,canal_2,canal_3,canal_4,canal_5,canal_6,canal_7,canal_8,timestamp,freq,aq'
    cols = cabecalho.split(',')
    
    # tempo no inicio da aquisição
    inicio = tempo()
    last_time = tempo()
    current_time = tempo() 
    
    if full:
        # definindo o inlet
        inlet = pylsl.stream_inlet(streams[0])
        
        # dataframe para guardar as os de varias divisoes
        df = pd.DataFrame(columns = cols)
        tam_divisao = int(tamanho_janela_segundos/divisao_minima)
        
        # file e objetos para salvar o sinal
        filename = os.path.join(path, 'tempo_real.csv')
        file_obj = open(filename, 'w')
        writer = csv.writer(file_obj, lineterminator = '\n')
        writer.writerow(cols)
        
        # escolhendo uma label
        freq, i = choose_freq_play_file(possible_freqs, i)  
        # definindo o tempo para duracao total
        tempo_0 = tempo()
        duracao_total = duration*(n+1)*(len(possible_freqs.keys()))
        
        
        while True:
            sample = []
            # caso passe a divisão minima, pega os resultados, processa e preve
            if current_time - last_time >= divisao_minima:
  
                # passa_lista, para salvar o index dessa iteração
                index_lista = passa_lista(index_lista, index, size)
                
                # reseta o valor index (controle)
                index = 0            
                
                # o tempo será tirado a partir de agora
                last_time = current_time
                
                # do dataframe que guarda uma janela, tira a mais antiga
                if len(index_lista) > tam_divisao:
                    df = df.iloc[index_lista[-tam_divisao-1]:].copy().reset_index(drop=True)
                
                # e poe a nova
                df = pd.concat((df,pd.DataFrame(signal, columns = cols)))
                
                # reseta o vetor de entrada
                signal = []
                # processa o sinal
                if len(index_lista) > tam_divisao:
                    sinal_processado = processa.processa_real_time(df.copy(), 
                                                                    tamanho_janela_segundos = tamanho_janela_segundos,
                                                                    high_Wn = high_Wn, 
                                                                    high_N = high_N,
                                                                    low_Wn=low_Wn, low_N = low_N, 
                                                                    hann=hann, hamm=hamm, fltop=fltop, wvlt=wvlt)
                
                    # transforma o output do processamento em um np array
                    dataset = np.array(sinal_processado)
                                    
                    # tenta prever o estado
                    saida = module.prever(rede, scaler, dataset)
                    
                    print(saida)
                    
                    # salva o output
                    output_lista = passa_lista(output_lista, saida, output_size)
                
            timestamp = inlet.pull_sample(sample)
            if timestamp[0]:
                current_time = tempo()
                index = index + 1          
                
                # append na lista
                signal.append([index]+timestamp[0]+[timestamp[1], freq, aq])
                
                # salva o sinal
                writer.writerow([index]+timestamp[0]+[timestamp[1], freq, aq])
            
            if current_time-inicio >= duration:
                inicio = tempo()
                freq,i = choose_freq_play_file(possible_freqs, i)
                aq+=1
                # redefinindo o inlet
                inlet = pylsl.stream_inlet(streams[0])
                
            if current_time - tempo_0 >= duracao_total:
                play_audio.play_file('Fim')
                break
        file_obj.close()
    file_dic = transform_real_time_files(path, possible_freqs)
    #file_dic = get_file_dic('')
    dataset, output = processa.roda_processamento(filenames_dic = file_dic,
                                                    source = 'tempo_real', tamanho_janela_segundos=tamanho_janela_segundos,
                                                    high_Wn = high_Wn, 
                                                    high_N = high_N,
                                                    low_Wn=low_Wn, low_N = low_N, 
                                                    hann=hann, hamm=hamm, fltop=fltop, wvlt=wvlt)
    
    module.prever_debug(rede, scaler, dataset, output)
    return file_dic
            
def choose_freq_play_file(possible_freqs, i):
    freq = list(possible_freqs.keys())[i]
    if i == len(possible_freqs)-1:
        i=0
    else:
        i+=1
    play_audio.play_file(freq)
    return freq, i

def transform_real_time_files(folder, possible_freqs):
    arquivo = os.path.join(folder, 'tempo_real.csv')
    freq_counts = {key:0 for key in possible_freqs.keys()}
    df = pd.read_csv(arquivo)
    file_dic = {}
    if not os.path.isdir(os.path.join(folder,'debug')):
        os.makedirs(os.path.join(folder,'debug'))
    for i in df['aq'].unique():
        temp = df[df['aq']== i].copy()
        if temp.shape[0] < 2500:
            continue
        freq = str(temp['freq'].unique()[0])
        assert temp['freq'].nunique() == 1
        file_label = str(freq)+'_'+str(freq_counts[freq])
        freq_counts[freq] +=1
        filename = os.path.join(folder,'debug', file_label+'hz.csv')
        file_dic[file_label] = filename
        temp.to_csv(filename, index=False)
    return file_dic
        
    
def run_forever(rede, scaler, module, streams, path, divisao_minima = 1, tamanho_janela_segundos = 4, size = 10, 
                high_Wn = 0.1, high_N = 8, low_Wn=0.1, 
                low_N = 8, hann=False, hamm=False, fltop=False, wvlt=False):    
    # vetor para guardar o sinal do open bci
    signal = []
    
    # index da posição de cada sinal na janela
    index = 0
    
    # lista que guardará os indexes maximos das janelas anteriores
    index_lista = []
    
    # quantos outputs são guardados
    output_size = 100
    # lista para guardar esses outputs
    output_lista = []
    
    # cabecalho do dataframe
    cabecalho = 'index_num,canal_1,canal_2,canal_3,canal_4,canal_5,canal_6,canal_7,canal_8,timestamp'
    cols = cabecalho.split(',')
    
    # dataframe para guardar as os de varias divisoes
    df = pd.DataFrame(columns = cols)
    tam_divisao = int(tamanho_janela_segundos/divisao_minima)
    
    # file e objetos para salvar o sinal
    filename = os.path.join(path, 'tempo_real.csv')
    file_obj = open(filename, 'w')
    writer = csv.writer(file_obj, lineterminator = '\n')
    writer.writerow(cols)
    
    # tempo no inicio da aquisição
    
    last_time = tempo()
    current_time = tempo()
    
    # definindo o inlet
    inlet = pylsl.stream_inlet(streams[0])
    
    while True:
        sample = []
        # caso passe a divisão minima, pega os resultados, processa e preve
        if current_time - last_time >= divisao_minima:
            # passa_lista, para salvar o index dessa iteração
            index_lista = passa_lista(index_lista, index, size)
            
            # reseta o valor index (controle)
            index = 0            
            
            # o tempo será tirado a partir de agora
            last_time = current_time
            
            # do dataframe que guarda uma janela, tira a mais antiga
            if len(index_lista) > tam_divisao:
                df = df.iloc[index_lista[-tam_divisao-1]:].copy().reset_index(drop=True)
            
            # e poe a nova
            df = pd.concat((df,pd.DataFrame(signal, columns = cols)))
            
            # reseta o vetor de entrada
            signal = []
            # processa o sinal
            if len(index_lista) > tam_divisao:
                sinal_processado = processa.processa_real_time(df.copy(), 
                                                                    tamanho_janela_segundos = tamanho_janela_segundos,
                                                                    high_Wn = high_Wn, 
                                                                    high_N = high_N,
                                                                    low_Wn=low_Wn, low_N = low_N, 
                                                                    hann=hann, hamm=hamm, fltop=fltop, wvlt=wvlt)
            
                # transforma o output do processamento em um np array
                dataset = np.array(sinal_processado)
                                
                # tenta prever o estado
                saida = module.prever(rede, scaler, dataset)
                
                print(saida)
                
                # salva o output
                output_lista = passa_lista(output_lista, saida, output_size)
            
        timestamp = inlet.pull_sample(sample)
        if timestamp[0]:
            current_time = tempo()
            index = index + 1
            
            
            # append na lista
            signal.append([index]+timestamp[0]+[timestamp[1]])
            
            # salva o sinal
            writer.writerow([index]+timestamp[0]+[timestamp[1]])
            
        
def salvar_objeto(objeto, folder, nome=''):
    if not nome:
        print('Forneça o nome do objeto')
        return
    print('Salvando objeto em ',os.path.join(folder, nome + '.pkl'))
    with open(os.path.join(folder, nome + '.pkl'), 'wb') as f:
        pickle.dump(objeto, f)
        
def carregar_objeto(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def run(treinar = True, adquirir = True, rede_path = '', file_dic = {}, 
        rede = '', divisao_minima = 1, tamanho_janela_segundos = 4,
        size = 80, hidden_layers = 3, debug=False, full_debug=False, search=False,
        open_bci=True, split = True, high_Wn = 0, high_N = 6, low_Wn=0.84, 
        low_N = 2, hann=False, hamm=True, fltop=False, wvlt=False):
    
    # seleção do método de classificação
    methods = [neural_network.rede_neural, svm_classifier.support_vector_machine]
    method = methods[0]
    
    if method == neural_network.rede_neural:
        print('Método de classificação escolhido: Rede Neural')
        module = neural_network
        hl_tuple = tuple(size for _ in range(hidden_layers+1))
        args = tuple((hl_tuple, split))
        nome = 'rede_neural'
    else:
        print('Método de classificação escolhido: SVM')
        module = svm_classifier
        args = tuple((split,))
        nome = 'svm'
    
    
    # tentando conectar ao OpenBCI
    if open_bci:
        try:
            streams = connect_controller()
            if not streams:
                return
        except:
            return
    else:
        streams = [1]
        
    # definindo os modos de operação
    if rede_path and rede:
        print('Você entrou com um path e uma rede, entre com apenas um')
        return
    
    elif treinar and adquirir:
        rede, scaler, path, file_dic = run_training(streams, method, args, tamanho_janela_segundos=4, divisao_minima=1,
                                                    high_Wn = high_Wn, 
                                                    high_N = high_N,
                                                    low_Wn=low_Wn,low_N = low_N, 
                                                    hann=hann, hamm=hamm, fltop=fltop, wvlt=wvlt)
        folder = os.path.dirname(path)
        salvar_objeto(rede, folder, nome='rede_neural')
        salvar_objeto(scaler, folder, nome='scaler')
    
    elif treinar and not adquirir and not rede_path:
        folder = 'C:\\Users\\Rafael\\Documents\\TCC\\BCI\\EEG_Data\\aquisicao_2017-11-07_10h30'
        folder = r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\aquisicao_2017-11-10_10h11'
        folder = r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\aquisicao_2017-11-11_16h14'
        folder = r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\aquisicao_2017-11-09_11h29'
        #folder = r'C:\Users\Rafael\Documents\TCC\BCI\EEG_Data\aquisicao_2017-11-13_14h19'
        #folder = 'C:\\Users\\Rafael\\Documents\\TCC\\BCI\\EEG_Data\\aquisicao_2017-11-11_17h01'
        #folder = 'C:\\Users\\Rafael\\Documents\\TCC\\BCI\\EEG_Data\\aquisicao_2017-11-12_22h43' # nelson
        #folder = ''
        
        if not file_dic:
            file_dic = get_file_dic(folder)
        folder = os.path.dirname(list(file_dic.values())[0])
        
        print('Processando dados...')
        dataset, output = processa.roda_processamento(filenames_dic = file_dic,
                                                      source = 'tempo_real', tamanho_janela_segundos=tamanho_janela_segundos,
                                                      high_Wn = high_Wn, 
                                                      high_N = high_N,
                                                      low_Wn=low_Wn, low_N = low_N, 
                                                      hann=hann, hamm=hamm, fltop=fltop, wvlt=wvlt)
        #pca.plot_pca(dataset, output)
        args = tuple((dataset, output)) + args
        
        # treinando o ML
        rede, scaler, result = method(*args)
        
        # salvando o objeto
        salvar_objeto(rede, folder, nome = nome)
        salvar_objeto(scaler, folder, nome='scaler')
   
    elif not treinar and not rede_path and not rede:
        print('Você escolheu entrar com uma rede treinada, mas não forneceu uma!')
        return
    elif (not treinar or not adquirir) and rede_path:
        path = setup()[0]
        folder = os.path.dirname(path)
        print(folder)
        try:
            rede = carregar_objeto(rede_path)
            scaler = carregar_objeto(rede_path.replace('rede_neural', 'scaler'))
        except:
            print('Caminho para a rede não encontrado!')
            return
    elif not treinar and rede:
        pass
    
    if debug:
        file_dic_real = run_debug_mode(rede, scaler, module, streams, folder, divisao_minima,
                                tamanho_janela_segundos, full=full_debug, 
                                high_Wn = high_Wn, 
                                high_N = high_N,
                                low_Wn=low_Wn, low_N = low_N, 
                                hann=hann, hamm=hamm, fltop=fltop, wvlt=wvlt)
    elif not debug and not search:
        print('Iniciando operação em tempo real...')
        print('Boa Sorte!')
        run_forever(rede, scaler, module, streams, folder, divisao_minima,
                                tamanho_janela_segundos,
                                high_Wn = high_Wn, 
                                high_N = high_N,
                                low_Wn=low_Wn,low_N = low_N, 
                                hann=hann, hamm=hamm, fltop=fltop, wvlt=wvlt)
    if debug and search:
        variables_dict = {
            'size': [50,100],
            'hidden_layers':[3,5],
            'high_Wn':[False,0.1,0.25,0.4],
            'high_N':[8,10],
            'low_Wn':[False,0.25,0.4,0.55,0.7,0.8,0.9],
            'low_N':[6,8],
            'hann':[True,False],
            'hamm':[True,False],
            'fltop':[False],
            'wvlt':[True,False]
                
        }
        results = grid_search(folder, file_dic, file_dic_real, variables_dict, split,
                    nome, tamanho_janela_segundos=tamanho_janela_segundos)
        results.to_csv(os.path.join(folder, 'results.csv'), index=False)
        
        """
        run(adquirir=False, debug=True, full_debug=False, search=True, open_bci=False)
        """
        
def run_datasets(user, dataset=2, divisao_minima = 1, tamanho_janela_segundos = 4,
                    size = 40, hidden_layers = 3, debug=True, search=False,
                    split = True, high_Wn = 0, high_N = 4, low_Wn=0, 
                    low_N = 5, hann=False, hamm=False, fltop=False, wvlt=False):
    
    # seleção do método de classificação
    methods = [neural_network.rede_neural, svm_classifier.support_vector_machine]
    method = methods[0]
    
    if method == neural_network.rede_neural:
        print('Método de classificação escolhido: Rede Neural')
        module = neural_network
        hl_tuple = tuple(size for _ in range(hidden_layers+1))
        args = tuple((hl_tuple, split))
        nome = 'rede_neural'
    else:
        print('Método de classificação escolhido: SVM')
        module = svm_classifier
        args = tuple((split,))
        nome = 'svm'
    
    # pegando o usuario e a pasta correspondente
    user = str(user).zfill(3)
    if dataset == 1:
        folder = os.path.join(os.getcwd(), 'download_eeg_data', user)
    elif dataset == 2:
        folder = os.path.join(os.getcwd(), 'download_eeg_data_2', 'data', user)
    file_dic = get_file_dic(folder, keyword='', dataset = dataset)
    folder = os.path.dirname(list(file_dic.values())[0])
        
    print('Processando dados...')
    dataset, output = processa.roda_processamento(filenames_dic = file_dic,
                                                  source = 'download', tamanho_janela_segundos=tamanho_janela_segundos,
                                                  high_Wn = high_Wn, 
                                                  high_N = high_N,
                                                  low_Wn=low_Wn, low_N = low_N, 
                                                  hann=hann, hamm=hamm, fltop=fltop, wvlt=wvlt)
    
    print('Dados Processados. Treinando a rede neural...')
    #pca.plot_pca(dataset, output)
    args = tuple((dataset, output)) + args
    
    # treinando o ML
    rede, scaler, result = method(*args)
    
    # salvando o objeto
    salvar_objeto(rede, folder, nome = nome)
    salvar_objeto(scaler, folder, nome='scaler')
    print('FIM')
        
    
if __name__ == "__main__":
    #run(adquirir=False, debug=True, full_debug=False, search=True, open_bci=False)
    #run()
    #for i in range(1):
    #    run_datasets(3)
    run()
    pass