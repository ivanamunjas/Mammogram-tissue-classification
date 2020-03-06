# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 22:44:11 2019

@author: ASUS
"""

import cv2
import radiomics
import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('labels.csv')
data = data.values

path = 'C:\\Users\\ASUS\\Desktop\\Diplomski\\novo_resenje\\clean_data\\'
        
files = [f for f in os.listdir(path)]

files.sort(key=lambda f: int(f.split('.')[0]))

f = np.zeros((len(files),33)) # features (obelezja)
y = np.zeros(len(files))

scaler = MinMaxScaler()

for ind in range(0, len(files)):
    
# =============================================================================
#     UCITAVANJE I FILTRIRANJE
# =============================================================================
    
    #im=cv2.imread('C:/Users/ASUS/Desktop/Diplomski/kropovane ociscene/{}'.format(ind)+'.jpg')
    im = cv2.imread(path + files[ind], 0)
    num, ext = files[ind].split('.')

    GLC = radiomics.glcm.RadiomicsGLCM(sitk.GetImageFromArray(np.array((im),dtype='uint8')),
                                           sitk.GetImageFromArray(np.where(im > 0, 1, 0)))
    GLC._initCalculation()
    GLC._calculateMatrix()
    GLC._calculateCoefficients()
    f[ind][0] = GLC.getJointEnergyFeatureValue()
    f[ind][1] = GLC.getContrastFeatureValue()
    f[ind][2] = GLC.getCorrelationFeatureValue()
    f[ind][3] = GLC.getSumSquaresFeatureValue()
    f[ind][4] = GLC.getIdFeatureValue()
    f[ind][5] = GLC.getSumAverageFeatureValue()
    f[ind][6] = GLC.getClusterTendencyFeatureValue()
    f[ind][7] = GLC.getSumEntropyFeatureValue()
    f[ind][8] = GLC.getJointEntropyFeatureValue()
    f[ind][9] = GLC.getDifferenceVarianceFeatureValue()
    f[ind][10] = GLC.getDifferenceEntropyFeatureValue()
    f[ind][11] = GLC.getImc1FeatureValue()
    f[ind][12] = GLC.getImc2FeatureValue()
    f[ind][13] = GLC.getAutocorrelationFeatureValue()
    f[ind][14] = GLC.getDifferenceAverageFeatureValue()
    f[ind][15] = GLC.getClusterShadeFeatureValue()
    f[ind][16] = GLC.getClusterProminenceFeatureValue()
    f[ind][17] = GLC.getMaximumProbabilityFeatureValue()
    f[ind][18] = GLC.getIdnFeatureValue()
    f[ind][19] = GLC.getIdmnFeatureValue()
    
    GLRM = radiomics.glrlm.RadiomicsGLRLM(sitk.GetImageFromArray(np.array((im),dtype='uint8')),
                                          sitk.GetImageFromArray(np.where(im > 0, 1, 0)))
    
    GLRM.calculateFeatures()
    
    f[ind][20] = GLRM.getShortRunEmphasisFeatureValue()
    f[ind][21] = GLRM.getLongRunEmphasisFeatureValue()
    f[ind][22] = GLRM.getGrayLevelNonUniformityFeatureValue()
    f[ind][23] = GLRM.getRunLengthNonUniformityFeatureValue()
    f[ind][24] = GLRM.getRunPercentageFeatureValue()
    
    hist, _ = np.histogram(np.array(im).flatten(), bins = 256)
    
#    plt.figure()
#    plt.hist(np.array(im).flatten(), bins = 256)
    
    hist_mean = 0
    hist_var = 0
    hist_std = 0
    hist_uniformity = 0
    hist_entropy = 0
    
    for i in range(0,len(hist)):
        hist_mean += i*hist[i]
        hist_var += np.power((i - hist_mean/(256*256)),2)*hist[i]/(256*256)
        hist_std = np.power(hist_var, 0.5)
        hist_uniformity += np.power(hist[i]/(256*256), 2)
    
    f[ind][25] = hist_mean/(256*256)
    f[ind][26] = hist_var
    f[ind][27] = hist_std
    f[ind][28] = hist_uniformity
    
    marg = np.histogramdd(np.ravel(im), bins = 256)[0]/im.size
    marg2 = list(filter(lambda p: p > 0, np.ravel(marg)))
    entropy = -np.sum(np.multiply(marg2, np.log2(marg2)))
    #print(entropy)
    
    marg2 = np.array(marg2)
    
    alpha = 3
    renyi_entropy = 1/(1 - alpha)*np.log2(np.sum(marg2**alpha))
    #print(renyi_entropy)
    
    alpha = 0.5
    beta = 0.7
    kapur_entropy = 1/(beta - alpha)*np.log2(np.sum((marg2**alpha)/(marg2**beta)))
    #print(kapur_entropy)
    
    yager_entropy = np.abs(np.sum(np.multiply(marg2,2) - np.ones(len(marg2))))
    #print(yager_entropy)
    
    f[ind][29] = entropy
    f[ind][30] = renyi_entropy
    f[ind][31] = kapur_entropy
    f[ind][32] = yager_entropy
    
    
#    print(hist_mean/(256*256))
#    print(hist_var)
#    print(hist_std)
#    print(hist_uniformity)
    
    if data[int(num) - 1][1] == 'G' or data[int(num) - 1][1] == 'D':
        y[ind] = 1
#    elif data[int(num) - 1][1] == 'D':
#        y[ind] = 2
    else:
        y[ind] = 0

    print(ind)

scaler.fit(f)
f = scaler.transform(f)

np.save('obelezja', f)
np.save('klase', y)