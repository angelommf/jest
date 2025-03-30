#Introductory file - It will be checked here the initial number of features we're dealing with and the number of healthy people and people with leukemya, from our training dataset
import numpy as np

ds=np.genfromtxt('Leukemya_data.csv',delimiter=',')
lb=np.genfromtxt('labels.csv')
ds_train=ds[:128]
lb_train=lb[:128]

n_features=len(ds[0])
healthy=0
has_leukemya=0
for i in lb_train:
	if i==1:
		healthy+=1
	else:
		has_leukemya+=1
        
print('Number of features: ',n_features)
print('\nIn our training dataset (with 128 patients) we have:')
print(healthy, ' healthy people.')
print(has_leukemya, ' pelople with leukemya.')

'''
Number of features:  186

In our training dataset (with 128 patients) we have:
111  healthy people.
17  pelople with leukemya.
'''