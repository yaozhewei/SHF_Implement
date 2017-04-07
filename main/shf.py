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


	def initialize_params(self, X):
		'''
		initialize the parameters (weights and bias). We use the transpose of W and b in order to simplify WX^T + b to XW + b.
		'''
		(m,n) = np.shape(X)
		(W,b) = ([0] * len(self.layers), [0] * len(self.layers))

		if self.activations[0] == 'ReLU':
			shift = 0.1
		else:
			shift = 0

		r = np.sqrt(6) / np.sqrt(n + self.layers[0] + 1)
		W[0] = 2 * r * np.random.rand(n, len(self.layers)) - r
		b[0] = np.zeros((1, len(self.layers))) + shift

		for i in range(1, len(self.layers)):
			if self.activations[i] == 'ReLU':
				shift = 0.1
			else:
				shift = 0

			r = np.sqrt(6) / np.sqrt(n + self.layers[i-1] + self.layers[i] + 1)
			W[i] = 2 * r * np.random.rand(n, len(self.layers)) - r
			b[i] = np.zeros((1, len(self.layers))) + shift
		self.numdata = m
		self.inputsize = n
		self.W = W
		self.b = b


	def packnet(self, W, b):
		'''
		pack W and b into a long vector
		'''
		theta = np.concatenate((W[0].flatten(1), b[0].flatten(1)))
		for i in range(1, len(self.layers)):
			theta = np.concatenate((theta, np.concatenate((W[i].flatten(1), b[i].flatten(1)))))

		return np.array(theta.reshape(len(theta), 1))

	def forward(self):
		pass

	def backward(self):
		pass

	def R_forward(self):
		pass

	def R_backward(self):
		pass

	def train(self, X, Y, testX, testY):
		'''
		The main function of train the NN.
		'''
		def setup():
			'''
			set up the NN. We shuffle the order of X in order to get minibatch for computing gradient and Gauss-Newton matrix. 
			'''
			self.initialize_params(X)
			index_X = list(range(self.numdata))
			np.random.shuffle(index_X)
			numgradbatches = len(index_X) / self.gradbatchsize
			numbatches = self.gradbatchsize / self.batchsize

			self.theta = self.packnet(self.W, self.b)

		setup()









		