#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:25:52 2021

@author: angelo
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ds=np.genfromtxt('Leukemya_data.csv',delimiter=',')

#feature selection #1 (correlation matrix)
df=pd.DataFrame(ds,columns=[i for i in range(len(ds[0]))]) 
corrMatrix=df.corr().abs() 

#the following two functions are just a helping hand in getting the most correlated features
def get_redundant_pairs(df):
    #Get diagonal and lower triangular pairs of correlation matrix
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

mcf=get_top_abs_correlations(corrMatrix,75) #75 most correlated features
ftd=[] #features to delete
ftk=[] #features to keep
for i in range(len(mcf)):
    if (mcf.index[i][0] not in ftd) and (mcf.index[i][1] not in ftk): #this loop will distinguish both features to keep and features to delete
        ftd.append(mcf.index[i][0])
        ftk.append(mcf.index[i][1])
    
ftd.sort(reverse=True)
for i in range(len(ftd)):
   ds=np.delete(ds,ftd[i],axis=1)
   
#feature selection #2 (numerical feature variance)
thresholder=VarianceThreshold(threshold=.95)
ds=thresholder.fit_transform(ds)
   
#feature selection #3 (PCA - Principal Component Analysis)
scaler=StandardScaler() #Standardize scale
scaler.fit(ds)
ds=scaler.transform(ds)

pca = PCA(n_components=9)
pca.fit(ds)
ds=pca.transform(ds)