# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:09:49 2017

@author: Rafael
"""

from sklearn.decomposition import PCA

def plot_pca(dataset, output):
    pca = PCA(n_components = 2)
    pca.fit(dataset)
    print(pca.explained_variance_ratio_)
    #print(pca.singular_values)