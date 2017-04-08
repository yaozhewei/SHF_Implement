import numpy as np
import load_data2
import shf

# prepare data
print('Preparing data...')
(trainX, testX, train_labels, test_labels) = load_data2.load_data()
(trainX, testX) = load_data2.standard(trainX, testX)

# parameter
_layers = [500, 10]
_dropout = [0.5, 0.5]
_maxepoch = 100
_gradbatchsize = 200
_batchsize = 20
_maxiter = 5
_activations = ['ReLU', 'softmax']
_cgdecay_ini = 0.5
_cgdecay_fnl = 0.99

print('Training...')
nn = shf.SHF(layers=_layers, dropout=_dropout, maxepoch=_maxepoch, gradbatchsize=_gradbatchsize, batchsize=_batchsize, maxiter=_maxiter, activations=_activations, cgdecay_ini = _cgdecay_ini, cgdecay_fnl = _cgdecay_fnl)
nn.train(trainX, train_labels, testX, test_labels)