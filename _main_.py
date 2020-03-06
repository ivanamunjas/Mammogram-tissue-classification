# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:07:55 2019

@author: ASUS
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

x = np.load('obelezja.npy')
y = np.load('klase.npy')


def plot_confusion(y_t, y_t_pred):
    data = confusion_matrix(y_t, y_t_pred)
    #classes = ['fatty', 'grandular', 'dense']
    classes = ['fatty', 'dense']
    df_cm = pd.DataFrame(data, columns = classes, index = classes)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
    

def classify(s, X_train, X_test, y_train, y_test):
    if s == 'SVM':
        # =====================================================================
        # SVM CLASSIFIER
        # =====================================================================
        
        print('------------------------------------------')
        print('SVM')
        
        svm_clf = svm.SVC(kernel = 'linear')
        
        sfs = SFS(svm_clf, k_features=10, forward=True, floating=False, scoring='accuracy', verbose=2, cv=5)
        sfs = sfs.fit(X_train, y_train)
        
        print('\nSequential Forward Selection (k=10):')
        print(sfs.k_feature_idx_)
        print('CV Score:')
        print(sfs.k_score_)
        
        best_features = list(sfs.k_feature_idx_)
        
        svm_clf.fit(X_train[:, best_features], y_train)
        
        y_train_pred = svm_clf.predict(X_train[:, best_features])
        print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))
#        print(precision_score(y_train, y_train_pred))
#        print(recall_score(y_train, y_train_pred))
#        print(f1_score(y_train, y_train_pred))
        
        plot_confusion(y_train, y_train_pred)
        
        
        y_test_pred = svm_clf.predict(X_test[:, best_features])
        print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))
#        print(precision_score(y_test, y_test_pred))
#        print(recall_score(y_test, y_test_pred))
#        print(f1_score(y_test, y_test_pred))
        
        plot_confusion(y_test, y_test_pred)
        
    elif s == 'KNN':
        # =====================================================================
        # KNN
        # =====================================================================
        
        print('------------------------------------------')
        print('KNN')
        
        knn_clf = KNeighborsClassifier(n_neighbors = 3)  
        
        sfs = SFS(knn_clf, k_features=10, forward=True, floating=False, scoring='accuracy', verbose=2, cv=3)
        sfs = sfs.fit(X_train, y_train)
        
        print('\nSequential Forward Selection (k=10):')
        print(sfs.k_feature_idx_)
        print('CV Score:')
        print(sfs.k_score_)
        
        best_features = list(sfs.k_feature_idx_)
        
        knn_clf.fit(X_train[:, best_features], y_train)
        
        y_train_pred = knn_clf.predict(X_train[:, best_features])
        print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))
        
        plot_confusion(y_train, y_train_pred)
        print(acc(y_train, y_train_pred))
        
        y_test_pred = knn_clf.predict(X_test[:, best_features])
        #print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))
        
        #print(confusion_matrix(y_test, y_test_pred)) 
        
        plot_confusion(y_test, y_test_pred)
        print(acc(y_test, y_test_pred))
        
    elif s == 'RF':
        # =============================================================================
        # Random forest
        # =============================================================================
        
        print('------------------------------------------')
        print('Random forest')
        
        rf_clf = DecisionTreeClassifier()
        
        sfs = SFS(rf_clf, 
                  k_features=5, 
                  forward=True, 
                  floating=False, 
                  scoring='accuracy',
                  verbose=2,
                  cv=10)
        
        sfs = sfs.fit(X_train, y_train)
        
        print('\nSequential Forward Selection (k=10):')
        print(sfs.k_feature_idx_)
        print('CV Score:')
        print(sfs.k_score_)
        
        best_features = list(sfs.k_feature_idx_)
        
        
        #rf_clf.fit(X_train[:, best_features], y_train)
        rf_clf.fit(X_train, y_train)
        
        #y_train_pred = rf_clf.predict(X_train[:, best_features])
        y_train_pred = rf_clf.predict(X_train)
        print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))
        
        plot_confusion(y_train, y_train_pred)
        #print(f1_score(y_train, y_train_pred))
        #y_test_pred = rf_clf.predict(X_test[:, best_features])
        y_test_pred = rf_clf.predict(X_test)
        print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))
        
        #print(confusion_matrix(y_test, y_test_pred)) 
        
        plot_confusion(y_test, y_test_pred)
        #print(f1_score(y_test, y_test_pred))
        
    elif s == 'NB':
        # =============================================================================
        # NAIVE BAYES
        # =============================================================================
        print('------------------------------------------')
        print('Naive Bayes')
        
        nb_clf = GaussianNB()
        
        sfs = SFS(nb_clf, 
                  k_features=5, 
                  forward=True, 
                  floating=False, 
                  scoring='f1',
                  verbose=2,
                  cv=3)
        
        sfs = sfs.fit(X_train, y_train)
        
        print('\nSequential Forward Selection (k=10):')
        print(sfs.k_feature_idx_)
        print('CV Score:')
        print(sfs.k_score_)
        
        best_features = list(sfs.k_feature_idx_)
        
        #svm_clf.fit(X_train, y_train)  
        #
        #y_pred = svm_clf.predict(X_test)
        
        nb_clf.fit(X_train[:, best_features], y_train)
        
        y_train_pred = nb_clf.predict(X_train[:, best_features])
        print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))
        
        plot_confusion(y_train, y_train_pred)
        print(f1_score(y_train, y_train_pred))
        
        y_test_pred = nb_clf.predict(X_test[:, best_features])
        print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))
        
        plot_confusion(y_test, y_test_pred)
        print(f1_score(y_test, y_test_pred))
        

    elif s == 'ANN':
        # =====================================================================
        #         ANN
        # =====================================================================
        classifier = Sequential()
        

        # Adding the input layer and the first hidden layer
        classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu', input_dim = 33))
        
        # Adding the second hidden layer
        classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))
        
        classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))
        
        
        
        # Adding the output layer
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        
        # Compiling the ANN
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        
        
        # Fitting the ANN to the Training set
        classifier.fit(X_train, y_train, batch_size = 10, epochs = 500)
        
        
        y_pred = classifier.predict(X_test)
        print(y_pred)
        
        for i in range(0,len(y_pred)):
            if y_pred[i] < 0.33:
                y_pred[i] = 0
            elif y_pred[i] > 0.33 and y_pred[i] < 0.66:
                y_pred[i] = 1
            else:
                y_pred[i] = 2
                
        
        y_train_pred = classifier.predict(X_train)
        
        for i in range(0,len(y_train_pred)):
            if y_train_pred[i] < 0.33:
                y_train_pred[i] = 0
            elif y_train_pred[i] > 0.33 and y_train_pred[i] < 0.66:
                y_train_pred[i] = 1
            else:
                y_train_pred[i] = 2
        
        print(acc(y_train, y_train_pred))
        plot_confusion(y_train, y_train_pred)
        
        print(acc(y_test, y_pred))
        plot_confusion(y_test, y_pred)


                
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

classifiers = ['SVM', 'KNN', 'RF', 'NB', 'ANN']

for c in classifiers:
    classify(c, X_train, X_test, y_train, y_test)