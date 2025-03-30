from random import randint
import numpy as np

#Undersampling when the minority class is 0.
def undersampling(train,labels):
    n=0
    Ind_leukemya=[]
    Ind_not=[]
    for i in range(len(labels)):
        if labels[i]==1:
            Ind_leukemya.append(n)
        else:
            Ind_not.append(n)
        n=n+1
    while len(Ind_not)>len(Ind_leukemya):
        r=randint(0,len(Ind_not)-1)
        del Ind_not[r]
    ind=0
    I=[]
    for i in range(len(labels)):
        I.append(ind)
        ind=ind+1
    for i in range(len(Ind_not)):
        I.remove(Ind_not[i])
    for i in range(len(Ind_leukemya)):
        I.remove(Ind_leukemya[i])
        
    X=np.delete(train,I,0)
    y=np.delete(labels,I,0)
    return [X,y]

#Oversampling when majority class is binary 0.    
def oversampling(train,labels): 
    n_positives=0
    positives = []
    for i in range(len(labels)):
        if labels[i] == 1:
            n_positives+=1
            positives.append(train[i])
    negatives = train.shape[0] - n_positives
    while(train.shape[0]<2*negatives):
        r=randint(0,len(positives)-1)
        new_positive = positives[r]
        train = np.vstack((train,new_positive))
        labels = np.append(labels,1)
        
    return[train,labels]
        