#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:16:29 2021

@author: angelo
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from feature_selection import ds
import numpy as np
import pandas as pd
import Sampling

ds1=ds[:128]
lb=np.genfromtxt('labels.csv')

ds_train, ds_test, lb_train, lb_test = train_test_split(ds1, lb, random_state=0, stratify=lb, test_size=0.2) #splitting in training and testing dataset and labels

new_ds=Sampling.oversampling(ds_train,lb_train) #oversampling pacients with leukemya

ds_train=new_ds[0] #more balanced ds
lb_train=new_ds[1]

clf=RandomForestClassifier(random_state = 0) #generate RF object
clf.fit(ds_train,lb_train)  #training the classifier
lb_pred=clf.predict(ds_test) #prediction of the labels of the 2nd chunk (test chunk)

print(confusion_matrix(lb_test, lb_pred))  #validation

'''
[[23  0]
 [ 0  3]]
'''
#23 True Negative e 3 True positive according to our model: seems legit

#Real prediction
lb_real=clf.predict(ds[128:])
lb_real=pd.DataFrame(lb_real)
lb_real.to_csv('Predicted_labels.csv',header=None,index=False)