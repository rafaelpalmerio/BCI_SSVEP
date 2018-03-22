# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:47:34 2017

@author: Rafael and Lucas
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import classification_report,confusion_matrix
import os
import random
import numpy as np

def save_result(conf, classif):
    if not os.path.isdir(os.path.join(os.getcwd(),'matrizes')):
        os.makedirs(os.path.join(os.getcwd(),'matrizes'))
    with open(os.path.join(os.getcwd(),'matrizes',str(random.randint(1,9999))+'.txt'), 'w') as f:
        f.write(str(conf) + '\n')
        f.write(str(classif))

def normalize(dataset):
    for index, line in enumerate(dataset):
        max_value = line.max()
        dataset[index] = line/max_value
    return dataset

def shuffle(dataset, output):
    assert len(dataset) == len(output)
    p = np.random.permutation(len(dataset))
    return dataset[p], output[p]


def rede_neural(final_dataset, final_output, hidden_layer_tuple, split):
    #final_dataset=final_dataset[:15]
    #final_output=final_output[:15]
    #final_dataset = normalize(final_dataset)
    
    # misturando as linhas de entrada e saida
    X, y = shuffle(final_dataset, final_output)
    
     
    if split:
        # separando em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y)    
    else:
        X_train, X_test = X,X
        y_train, y_test = y,y
    
    
    # inicializando o scaler do input
    scaler = StandardScaler()
    
    # fitando a entrada pelo scaler
    scaler.fit(X_train)
    
    # agora aplicamos o scaler para o os X
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #X_train = scale(X_train)
    #X_test= scale(X_test)
    
    # incicilizando a rede    
    rede = MLPClassifier(hidden_layer_sizes=hidden_layer_tuple, max_iter = 9999999, solver='lbfgs')
    
    # treinando o modelo
    rede.fit(X_train,y_train)
    
    # classificando as linhas de teste
    predictions = rede.predict(X_test)
    
    # validação do treinamento
    conf_matrix = confusion_matrix(y_test,predictions)
    report = classification_report(y_test,predictions)
    
    save_result(conf_matrix, report)
    
    print(conf_matrix)
    print(report)
    return rede, scaler, get_rate(conf_matrix)

def prever(rede, scaler, dataset):
    # normalizando a amostra
    X = scaler.transform(dataset)
    
    # prevendo a amostra
    return rede.predict(X)

def prever_debug(rede, scaler, dataset, output):
    # normalizando a amostra
    X = scaler.transform(dataset)
    
    # prevendo os resultados
    predictions = rede.predict(X)
    
    # comparando com os resultados esperados
    conf_matrix = confusion_matrix(output,predictions)
    report = classification_report(output,predictions)
    print(conf_matrix)
    print(report)
    save_result(conf_matrix, report)
    return get_rate(conf_matrix)
    
def get_rate(matrix):
    return float(matrix.trace())/matrix.sum()