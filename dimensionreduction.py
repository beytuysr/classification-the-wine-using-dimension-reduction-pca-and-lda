# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:37:43 2019
dimension reduction pca
@author: Beytu
"""

#1.import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. data preprocessing

#2. import data
veriler = pd.read_csv('Wine.csv')

X=veriler.iloc[:,0:13].values
Y=veriler.iloc[:,13].values

#split the data train and test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

#data normalization
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train2=pca.fit_transform(X_train)
X_test2=pca.transform(X_test)


#before pca
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#after pca
classifier2=LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

#prediction
y_pred=classifier.predict(X_test)
y_pred2=classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred) 
print('real/without pca')
print(cm)  

cm1=confusion_matrix(y_test,y_pred2) 
print('real/with pca')
print(cm1)  

cm2=confusion_matrix(y_pred,y_pred2) 
print('without pca and with pca')
print(cm2)  


#lda linear discriminat analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)
X_train_lda=lda.fit_transform(X_train,y_train)
X_test_lda=lda.transform(X_test)

#after lda
classifier_lda=LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

#predict
y_pred_lda=classifier_lda.predict(X_test_lda)

cm3=confusion_matrix(y_test,y_pred_lda) 
print('lda/real')
print(cm3)  
