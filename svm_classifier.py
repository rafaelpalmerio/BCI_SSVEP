# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 19:25:19 2017

@author: Rafael
"""

from sklearn import svm
from neural_network import shuffle, get_rate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import classification_report,confusion_matrix

import numpy as np

def support_vector_machine(final_dataset, final_output, split):    
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
    
    svm_obj = svm.SVC()
    
    # treinando o modelo
    svm_obj.fit(X_train,y_train)
    
    # classificando as linhas de teste
    predictions = svm_obj.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test,predictions)
    print(conf_matrix)
    print(classification_report(y_test,predictions))
    return svm_obj, scaler, get_rate(conf_matrix)

def prever(svm_obj, scaler, dataset):
    # normalizando a amostra
    X = scaler.transform(dataset)
    
    # prevendo a amostra
    return svm_obj.predict(X)

def prever_debug(rede, scaler, dataset, output):
    # normalizando a amostra
    X = scaler.transform(dataset)
    
    # prevendo os resultados
    predictions = rede.predict(X)
    
    conf_matrix = confusion_matrix(output,predictions)
    print(conf_matrix)
    print(classification_report(output,predictions))
    return get_rate(conf_matrix)