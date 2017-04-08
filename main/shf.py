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
				 activations=None,
				 cgdecay_ini = 0.5,
				 cgdecay_fnl = 0.99):

		self.layers = layers
		self.dropout = dropout
		self.maxepoch = maxepoch
		self.gradbatchsize = gradbatchsize
		self.batchsize = batchsize
		self.maxiter = maxiter
		self.activations = activations
		self.cgdecay_ini = cgdecay_ini
		self.cgdecay_fnl = cgdecay_fnl


	def initialize_params(self, X):
		'''
		initialize the parameters (weights and bias).
		'''
		(m,n) = np.shape(X)
		#print(m,n)
		(W,b) = ([0] * len(self.layers), [0] * len(self.layers))

		if self.activations[0] == 'ReLU':
			shift = 0.1
		else:
			shift = 0

		r = np.sqrt(6) / np.sqrt(n + self.layers[0] + 1)
		W[0] = 2 * r * np.random.rand(n, self.layers[0]) - r
		b[0] = np.zeros((1, self.layers[0])) + shift

		for i in range(1, len(self.layers)):
			if self.activations[i] == 'ReLU':
				shift = 0.1
			else:
				shift = 0

			r = np.sqrt(6) / np.sqrt(n + self.layers[i-1] + self.layers[i] + 1)
			W[i] = 2 * r * np.random.rand(self.layers[i-1], self.layers[i]) - r
			b[i] = np.zeros((1, self.layers[i])) + shift
		self.numdata = m
		self.inputsize = n
		self.W = W
		self.b = b


	def packnet(self, W, b):
		'''
		pack W and b into a long vector
		'''
		#print(W[0].shape)
		theta = np.concatenate((W[0].flatten(1), b[0].flatten(1)))
		for i in range(1, len(self.layers)):
			theta = np.concatenate((theta, np.concatenate((W[i].flatten(1), b[i].flatten(1)))))
		theta = theta.reshape(len(theta),1)
		#print(len(theta))
		return np.array(theta)


	def unpacknet(self, theta):
		index = 0
		(W, b) = ([0] * len(self.layers), [0] * len(self.layers))
		num_w = self.inputsize * self.layers[0]
		#print(num_w)
		num_b = self.layers[0]
		W[0] = np.reshape(theta[index : index + num_w], (self.inputsize, self.layers[0]), order='F')
		index += num_w
		b[0] = np.reshape(theta[index : index + num_b], (1, self.layers[0]), order = 'F')
		index += num_b

		W[0] = np.array(W[0])
		b[0] = np.array(b[0])

		for i in range(len(self.layers) - 1):
			num_w = self.layers[i] * self.layers[i+1]
			num_b = self.layers[i+1]
			W[i+1] = np.reshape(theta[index : index + num_w], (self.layers[i], self.layers[i+1]), order = 'F')
			index += num_w
			b[i+1] = np.reshape(theta[index : index + num_b], (1, self.layers[i+1]), order = 'F')
			index += num_b

			W[i+1] = np.array(W[i+1])
			b[i+1] = np.array(b[i+1])			

		return (W, b)


	def forward(self, theta, X, batchtype = 'obj'):
		'''
		FeedForwaed for the NN.
		obj: objective (gradient)   GV: Gauss-Newton vector
		'''
		(batchsize, n) = np.shape(X)
		(W, b) = self.unpacknet(theta)

		acts = [0] * (len(self.layers) + 1)
		acts[0] = np.concatenate((X, np.ones((batchsize, 1))), 1) 

		for i in range(len(self.layers)):
			preacts = np.dot(acts[i], np.concatenate((W[i], b[i])))

			if self.activations[i] == 'linear':
				acts[i+1] = np.concatenate((preacts, np.ones((batchsize, 1))), 1)
			elif self.activations[i] == 'logistic':
				acts[i+1] = np.concatenate(( 1 / (1 + np.exp(-preacts)), np.ones((batchsize, 1))), 1)
			elif self.activations[i] == 'ReLU':
				acts[i+1] = np.concatenate((preacts * (preacts > 0), np.ones((batchsize, 1))), 1)
			elif self.activations[i] == 'softmax':
				acts[i+1] = np.exp(preacts - preacts.max(1).reshape(batchsize, 1))
				acts[i+1] = np.concatenate((acts[i+1] / acts[i+1].sum(1).reshape(batchsize, 1), np.ones((batchsize, 1))), 1)

			if i == len(self.layers) - 2:
				if batchtype == 'obj':
					acts[i+1][:, :-1] = acts[i+1][:, :-1] * self.obj_dropmask
				elif batchtype == 'GV':
					acts[i+1][:, :-1] = acts[i+1][:, :-1] * self.GV_dropmask

		return acts
			#print(acts[i+1].shape)
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
			numgradbatches = len(index_X) // self.gradbatchsize
			numbatches = self.gradbatchsize // self.batchsize

			self.theta = self.packnet(self.W, self.b)
			#print(self.theta.shape)
			numparameters = len(self.theta)
			self.mask = np.ones((numparameters, 1))
			(maskW, maskB) = self.unpacknet(self.mask)
			for i in range(len(self.layers)):
				maskB[i][:] = 0
			self.mask = self.packnet(maskW, maskB)
			decrease = 0.99
			boost = 1 / decrease
			ch = np.zeros((numparameters, 1))
			f_decay = 1
			#print(self.mask.shape)
			del maskW
			del maskB

			return (index_X, numgradbatches, numbatches, decrease, boost, ch, f_decay)

		# main
		(index_X, numgradbatches, numbatches, decrease, boost, ch, f_decay) = setup()
		count = 0
		self.err_record =[]

		for epoch in range(self.maxepoch):

			if epoch > 0:
				f_decay *= 0.998
				if self.cgdecay_ini < self.cgdecay_fnl:
					self.cgdecay_ini = np.minimum(1.01 * self.cgdecay_ini, self.cgdecay_fnl)

			self.epoch = epoch

			for minigradbatch in range(numgradbatches):
				gradbatchX = X[index_X[minigradbatch::numgradbatches]]
				gradbatchY = Y[index_X[minigradbatch::numgradbatches]]

				if self.dropout[0] > 0:
					gradbatchX = gradbatchX * ( np.random.rand(gradbatchX.shape[0], gradbatchX.shape[1]) > self.dropout[0])

				batchX = gradbatchX[np.mod(count, numbatches) :: numbatches]
				batchY = gradbatchY[np.mod(count, numbatches) :: numbatches]

				if self.dropout[1] > 0:
					self.obj_dropmask = np.random.rand(gradbatchX.shape[0], self.layers[-2]) > self.dropout[1]
					self.GV_dropmask = self.obj_dropmask[np.mod(count, numbatches)::numbatches]
				# main function
				acts = self.forward(self.theta, gradbatchX, batchtype='obj')











		