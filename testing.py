#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:21:04 2018

@author: thinkpad
"""

import ppca 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import LogisticRegression

model = ppca.PPCA(latent_dim=128,n_iter=200)

import pickle
import gzip
import time
import collections


f = gzip.open('mnist.pkl.gz', 'r')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

train_data, train_labels = train_set
valid_data, valid_labels = valid_set
test_data, test_labels = test_set

d = train_data.shape[1]

model1 = ppca.PPCA(latent_dim=128,n_iter=100)
start=time.time()
model1.fit(train_data)
print("done in %f"%(time.time()-start))



model2 = PCA(n_components = 128)
start=time.time()
model2.fit(train_data)
print("done in %f"%(time.time()-start))

X_ppca = model1.transform(train_data)
X_pca = model2.transform(train_data)

logreg1 = LogisticRegression(C=2)
logreg2 = LogisticRegression(C=2)

logreg1.fit(X_ppca,train_labels)
logreg2.fit(X_pca,train_labels)


X_ppca_t = model1.transform(test_data)
X_pca_t = model2.transform(test_data)

logreg1.score(X_ppca_t,test_labels)
logreg2.score(X_pca_t,test_labels)
