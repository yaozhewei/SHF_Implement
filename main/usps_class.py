import numpy as np
from matplotlib import pyplot as plt
import load_data
import shf

# prepare data
print('Preparing data...')
(trainX, testX, train_labels, test_labels) = load_data.load_data()
(trainX, testX) = load_data.normization_data(trainX, testX)

# parameter
_layers = [500, 500, 10]
_dropout = [0.5, 0.5]
_maxepoch = 500
_gradbatchsize = 1000
_batchsize = 100
_maxiter = 5
_activations = ['ReLU', 'ReLU', 'softmax']
_cgdecay_ini = 0.5
_cgdecay_fnl = 0.99
_objfun = 'softmax-entropy'
_weight_cost = 2e-5
_damp = 1

print('Training...')
nn = shf.SHF(layers=_layers, dropout=_dropout, maxepoch=_maxepoch, gradbatchsize=_gradbatchsize, batchsize=_batchsize, maxiter=_maxiter, activations=_activations, cgdecay_ini = _cgdecay_ini, cgdecay_fnl = _cgdecay_fnl, objfun = _objfun, weight_cost = _weight_cost, damp = _damp)
(trainerr, testerr, damping) = nn.train(trainX, train_labels, testX, test_labels)

x_axis = list(range(_maxepoch))
plt.plot(x_axis, trainerr)
plt.plot(x_axis, testerr)
plt.show()

plt.plot(x_axis, damping)
plt.show()