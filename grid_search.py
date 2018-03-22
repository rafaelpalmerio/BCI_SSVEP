# -*- coding: utf-8 -*-

import itertools
import processa_dados_eeg_neural_network_real_time as processa
from multiprocessing import Pool
import pandas as pd
import neural_network
import svm_classifier
import sys
#sys.modules['__main__'].__file__ = 'ipython'

def grid_search(folder, file_dic, file_dic_real, variables_dict, split, nome, tamanho_janela_segundos=4):
    # pegando a posição das variaveis no dicionario
    pos_dict = {}
    try:
        keys = list(variables_dict.keys())
        pos_dict['size'] =  keys.index('size')
        pos_dict['hidden_layers'] =  keys.index('hidden_layers')
        pos_dict['high_Wn'] =  keys.index('high_Wn')
        pos_dict['high_N'] =  keys.index('high_N')
        pos_dict['low_Wn'] =  keys.index('low_Wn')
        pos_dict['low_N'] =  keys.index('low_N')
        pos_dict['hann'] =  keys.index('hann')
        pos_dict['hamm'] =  keys.index('hamm')
        pos_dict['fltop'] =  keys.index('fltop')
        pos_dict['wvlt'] =  keys.index('wvlt')
    except:
        print('Pase todas as variávels no dicionário')
        return
    
    params = list(itertools.product(*variables_dict.values()))
    print('Combinações de parâmetros do grid search: ', len(params))
    p = Pool()
    
    # iterando nas combinações de param
    print('Executando Grid Search...')
    
    results = p.map(apply_params, [(file_dic, file_dic_real, param_comb, order,
                                    pos_dict, tamanho_janela_segundos, nome, split)
                                    for order, param_comb in enumerate(params)])
    p.close()
    p.join()
    
    # concatenado resultados 
    final = pd.concat(results, ignore_index=True)
    
    print('Grid Search finalizado.')
    print(final)
    return final

def apply_params(params):
    file_dic, file_dic_real, param_comb, order, pos_dict, tamanho_janela_segundos, nome, split = params
    
    #print('\n\n\n\n\n\n', order,file_dic, '\n\n\n\n\n\n')
    print('\n\n\n\n\n\n', order, '\n\n\n\n\n\n')
    # seleção do método de classificação    
    if nome == 'rede_neural':
        method = neural_network.rede_neural
        module = neural_network
    else:
        method = svm_classifier.support_vector_machine
        module = svm_classifier
        
    # pegando cada param
    result = {variable: param_comb[pos_dict[variable]] for variable in pos_dict.keys() }
    
    # rodando o processamento com os parametros espeficados
    dataset_treino, output_treino = processa.roda_processamento(filenames_dic = file_dic,
                                                  source = 'tempo_real', tamanho_janela_segundos=tamanho_janela_segundos,
                                                  high_Wn = result['high_Wn'], 
                                                  high_N = result['high_N'],
                                                  low_Wn=result['low_Wn'],
                                                  low_N = result['low_N'], 
                                                  hann=result['hann'], 
                                                  hamm=result['hamm'],
                                                  fltop=result['fltop'], 
                                                  wvlt=result['wvlt'])
    
    # contruindo o hl_tuple da rede neural
    hl_tuple = tuple(result['size'] for _ in range(result['hidden_layers']))
    args = tuple((hl_tuple, split))
    args = tuple((dataset_treino, output_treino)) + args
    rede, scaler, tr_result = method(*args)
    
    
    # pegando o dataset e o output do aquisição de teste
    dataset_real, output_real = processa.roda_processamento(filenames_dic = file_dic_real,
                                                      source = 'tempo_real', tamanho_janela_segundos=tamanho_janela_segundos,
                                                      high_Wn = result['high_Wn'], 
                                                      high_N = result['high_N'],
                                                      low_Wn=result['low_Wn'],
                                                      low_N = result['low_N'], 
                                                      hann=result['hann'], 
                                                      hamm=result['hamm'],
                                                      fltop=result['fltop'], 
                                                      wvlt=result['wvlt'])
    
    test_result = module.prever_debug(rede, scaler, dataset_real, output_real)
    result['train_result'] = tr_result
    result['test_result'] = test_result    
    
    return pd.DataFrame([result], columns=result.keys())