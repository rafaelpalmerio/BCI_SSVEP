# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 19:25:19 2017

@author: Rafael
"""

from sklearn import svm
from neural_network import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import classification_report,confusion_matrix

import numpy as np

def support_vector_machine(final_dataset, final_output, hidden_layer_tuple):    
    # misturando as linhas de entrada e saida
    X, y = shuffle(final_dataset, final_output)
    
    # separando em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    #X_train, X_test = X,X
    #y_train, y_test = y,y
    
    
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
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    return svm_obj, scaler

def prever(svm_obj, scaler, dataset):
    # normalizando a amostra
    X = scaler.transform(dataset)
    
    # prevendo a amostra
    return svm_obj.predict(X)