# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:13:34 2017

@author: Rafael
"""

"""kappa['frequencia_olhos'][((kappa['frequencia_olhos']<8) & (kappa['frequencia_olhos']>6)) | ((kappa['frequencia_olhos']<10) & (kappa['frequencia_olhos']>8)) | ((kappa['frequencia_olhos']<12) & (kappa['frequencia_olhos']>10)) | ((kappa['frequencia_olhos']<14) & (kappa['frequencia_olhos']>12)) | ((kappa['frequencia_olhos']<16) & (kappa['frequencia_olhos']>14))]"""
freqs = [7,9,11,13,15]
def filtra_apply(row):
    for freq in freqs:
        if row['frequencia_olhos']<freq+1 and row['frequencia_olhos']>freq-1:
            return freq
    return None

#kappa['trial'] = kappa.apply(filtra_apply, axis=1)

def recall_precision(matrix):
    size = matrix.shape[0]
    df = pd.DataFrame(matrix)
    score_df = pd.DataFrame(columns=['precision', 'recall', 'support'])
    for line in range(size):
        row = {}
        row['precision'] = df[line].iloc[line]/df[line].sum()
        row['recall'] = df[line].iloc[line]/df.iloc[line].sum()
        row['support'] = df.iloc[line].sum()
        score_df = score_df.append(row, ignore_index=True)
        
    avg_total = score_df.mean()
    avg_total['support'] = int(score_df['support'].sum())
    return score_df, avg_total