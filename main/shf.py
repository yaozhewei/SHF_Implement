import numpy as np

class SHF(object):
	#  NN with SHF

	def __init__(self, 
				 layers=None, 
				 dropout=None, 
				 maxepoch=1000, 
				 gradbatchsize=1000, 
				 batchsize=100, 
				 maxiter=5, 
				 activations=None):
		self.layers = layers
		self.dropout = dropout
		self.maxepoch = maxepoch
		self.gradbatchsize = gradbatchsize
		self.batchsize = batchsize
		self.maxiter = maxiter
		self.activations = activations


	def initialize_params(self):
		pass

	def forward(self):
		pass

	def backward(self):
		pass

	def R_forward(self):
		pass

	def R_backward(self):
		pass

	def train(self, trainX, train_labels, testX, test_Labels):
		'''
		The main function of train the NN.
		'''
		def setup():
			pass

		setup()









		