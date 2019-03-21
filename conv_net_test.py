# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:46:03 2019

@author: gela
"""

import numpy as np
from Nafo.cnn_theano import Convoutional_Neural_net as cnn
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as T

data = np.array(pd.read_csv('train.csv')).astype(np.float32)

Y_train = data[:1000,0].astype(np.int32)
X_train = data[:1000, 1:]

X_reshaped = np.float32(X_train.reshape(1000, 1, 28, 28)/255)

'''
plt.imshow(X_reshaped[3, 0, :, :], cmap = 'Greys')
plt.show()
'''

gela = cnn(CNN = [((30, 1, 5, 5), T.nnet.relu, (2,2)),
                  ((10, 30, 5, 5), T.nnet.relu, (2,2))
                 ],
           ANN = [(('x', 800), T.nnet.relu),
                  ((800, 800), T.nnet.relu),
                  ((800, 10), T.nnet.softmax)],
           shape_x = 28)


pred = gela.fit(X_reshaped, Y_train, lr_cnn = 5e-15,
                lr_ann = 5e-9, 
                mu_ann = 0.7,
                mu_cnn = 0.7,
                epoch = 201,
                print_period = 10,
                batch_size = 100)


cnn_weights, ann_weights = gela.get_weights() #this one line really does stuff tho







'''
i = 0

for i in range(len(gela.ws_theano_cnn)):
    filt, bias = gela.ws_theano_cnn[i]
    filt_np = filt.eval()
    bias_np = bias.eval()
    
    cnn_weights.append(filt_np)
    cnn_bias.append(bias_np)
    

ann_weights = []
ann_bias = []


for j in range(len(gela.ws_theano_ann)):
    weight, biass = gela.ws_theano_ann[j]
    weight_np = weight.eval()
    biass_np = biass.eval()

    ann_weights.append(weight_np)
    ann_bias.append(biass_np)


cnn_weights[0] = cnn_weights[0].reshape(250)
cnn_weights[1] = cnn_weights[1].reshape(2500)

'''

'''
cnn_weights = np.array(cnn_weights)
cnn_bias = np.array(cnn_bias)
ann_weights = np.array(ann_weights)
ann_bias = np.array(ann_bias)


filter1 = cnn_weights[0]
filter2 = cnn_weights[1]
    
cnnbias1 = cnn_bias[0]
cnnbias2 = cnn_bias[1]

ann_w1 = ann_weights[0]
ann_w2 = ann_weights[1]
ann_w3 = ann_weights[2]

ann_bias1 = ann_weights[0]
ann_bias2 = ann_weights[1]
ann_bias3 = ann_weights[2]


filter1.tofile('filter1.dat')
filter2.tofile('filter2.dat')

cnnbias1.tofile('cnnbias1.dat')
cnnbias2.tofile('cnnbias2.dat')

ann_w1.tofile('annw1.dat')
ann_w2.tofile('annw2.dat')
ann_w3.tofile('annw3.dat')

ann_bias1.tofile('abias1.dat')
ann_bias2.tofile('abias2.dat')
ann_bias3.tofile('abias3.dat')
'''




