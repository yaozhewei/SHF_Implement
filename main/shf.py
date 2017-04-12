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
				 cgdecay_fnl = 0.99,
				 objfun='softmax-entropy',
				 weight_cost = 2e-5,
				 damp = 0.1,
				 p_i=0.5,
				 p_f=0.99):

		self.layers = layers
		self.dropout = dropout
		self.maxepoch = maxepoch
		self.gradbatchsize = gradbatchsize
		self.batchsize = batchsize
		self.maxiter = maxiter
		self.activations = activations
		self.cgdecay_ini = cgdecay_ini
		self.cgdecay_fnl = cgdecay_fnl
		self.objfun = objfun
		self.weight_cost = weight_cost
		self.damp = damp
		self.p_i = p_i
		self_p_f = p_f

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


	def forward(self, theta, X, batchtype='obj', test=False):
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

			if i == len(self.layers) - 2 and not test:
				if batchtype == 'obj':
					acts[i+1][:, :-1] = acts[i+1][:, :-1] * self.obj_dropmask
				elif batchtype == 'GV':
					acts[i+1][:, :-1] = acts[i+1][:, :-1] * self.GV_dropmask

		return acts
	

	def objective(self, theta, Y, acts):
		'''
		Objective function
		'''
		(batchsize, n) = np.shape(acts[0])
		objfunction = self.objfun

		if objfunction == 'MSE':
			out = 0.5 * np.sum((acts[-1][:,:-1] - Y)**2) / batchsize
		elif objfunction == 'cross-entropy':
			out = -np.sum(Y * np.log(acts[-1][:,:-1] + 1e-20) + (1 - Y) * np.log(1 - acts[-1][:,:-1] + 1e-20)) / batchsize
		elif objfunction == 'softmax-entropy':
			out = -np.sum(Y * np.log(acts[-1][:,:-1] + 1e-20)) / batchsize

		out += 0.5 * self.weight_cost * ((self.mask * theta) ** 2).sum()
		return out


	def backward(self, Y, acts):
		'''
		Calculate gradient and precondition of CG 
		'''
		(batchsize, n) = np.shape(acts[0])
		dW =[0] * len(self.layers)
		db = [0] * len(self.layers)
		dW2 = [0] * len(self.layers)
		db2 = [0] * len(self.layers)

		Lx = (acts[-1][:,:-1] - Y) / batchsize

		for i in range(len(self.layers)-1, -1, -1):
			delta = np.dot(acts[i].T, Lx)
			dW[i] = delta[:-1, :]
			db[i] = delta[-1, :]

			delta2 = batchsize * np.dot(acts[i].T**2, Lx**2)
			dW2[i] = delta2[:-1,:]
			db2[i] = delta2[-1,:]

			if i > 0:
				if self.activations[i-1] == 'linear':
					Lx = np.dot(Lx, np.concatenate((self.W[i], self.b[i])).T)
				elif self.activations[i-1] == 'logistic':
					Lx = np.dot(Lx, np.concatenate((self.W[i], self.b[i])).T) * (acts[i] * (1 - acts[i]))
				elif self.activations[i-1] == 'ReLU':
					Lx = np.dot(Lx, np.concatenate((self.W[i], self.b[i])).T) * (acts[i] > 0)
				Lx = Lx[:,:-1]
		grad = self.packnet(dW, db)
		grad = grad + self.weight_cost * (self.mask * self.theta)
		precon = self.packnet(dW2, db2)
		precon = (precon + np.ones((len(grad), 1)) * self.damp + self.weight_cost * self.mask)**(3.0/4.0)

		return (grad, precon)	


	def R_forward(self, acts, v):
		'''
		R-operator forward part
		'''
		(batchsize, n) = np.shape(acts[0])
		W = self.W
		b = self.b

		R = [0] * (len(self.layers) + 1)
		R[0] = np.zeros((batchsize, self.inputsize + 1))
		(VW, Vb) = self.unpacknet(v)

		for i in range(len(self.layers)):
			bz = np.zeros(np.shape(b[i]))
			R[i+1] = np.concatenate((np.dot(R[i], np.concatenate((W[i], bz))), np.zeros((batchsize, 1))), 1) + np.concatenate((np.dot(acts[i], np.concatenate((VW[i], Vb[i]))), np.ones((batchsize, 1))), 1)

			if self.activations[i] == 'logistic':
				R[i+1] = R[i+1] * (acts[i+1] * (1 - acts[i+1]))
			elif self.activations[i] == 'ReLU':
				R[i+1] = R[i+1] * (acts[i+1] > 0)
			elif self.activations[i] == 'softmax':
				R[i+1] = R[i+1] * acts[i+1] - acts[i+1] * (acts[i+1][:,:-1] * R[i+1][:,:-1]).sum(1).reshape(batchsize, 1)

		return R[-1]

	def R_backward(self, acts, R, v):
		"""
		Backward for computing Jv 
		"""
		(batchsize, n) = np.shape(acts[0])
		dVW = [0] * len(self.layers)
		dVb = [0] * len(self.layers)

		RIx = R[:,:-1] / batchsize

		for i in range(len(self.layers) - 1, -1, -1):
			delta = np.dot(acts[i].T, RIx)
			dVW[i] = delta[:-1,:]
			dVb[i] = delta[-1,:]

			if i > 0:
				if self.activations[i-1] == 'linear':
					RIx = np.dot(RIx, np.concatenate((self.W[i], self.b[i])).T)
				elif self.activations[i-1] == 'logistic':
					RIx = np.dot(RIx, np.concatenate((self.W[i], self.b[i])).T) * (acts[i] * (1 - acts[i]))
				elif self.activations[i-1] == 'ReLU':
					RIx = np.dot(RIx, np.concatenate((self.W[i], self.b[i])).T) * (acts[i] > 0)
			RIx = RIx[:,:-1]

		gv = self.packnet(dVW, dVb)
		gv = gv + self.weight_cost * (self.mask * v)
		gv = gv + self.damp * v

		return gv

	def computeGV(self, X, v, acts=None):
		'''
		compute gauss-newton vector: G*V
		'''
		gv = self.R_backward(acts, self.R_forward(acts, v), v)

		return gv

	def conjgrad(self, X, b, x0, acts, precon, maxiter):
		'''
		CG for solving Gx = -grad.
		'''
		IS = [0] * maxiter
		XS = [0] * maxiter
		r = self.computeGV(X, x0, acts) - b
		y = r / precon
		p = -y
		x = x0
		delta_new = (r * y).sum()

		for i in range(maxiter):

			Ap = self.computeGV(X, p, acts)
			pAp = (p * Ap).sum()
			alpha = delta_new / pAp
			x = x + alpha * p
			r_new = r + alpha * Ap
			y_new = r_new / precon
			delta_old = delta_new
			delta_new = (r_new * y_new).sum()
			beta = delta_new / delta_old
			p = -y_new + beta * p
			r = r_new
			y = y_new

			IS[i] = i
			XS[i] = x

		return (XS, IS)

	def results(self, X, Y, testX, testY):
		'''
		compute the result and error
		'''
		acts = self.forward(self.theta, X, test=True)
		train_obj = self.objective(self.theta, Y, acts)
		yhat = np.argmax(acts[-1][:,:-1], 1)
		train_L = np.argmax(Y, 1)
		train_err = np.mean(train_L != yhat)

		acts = self.forward(self.theta, testX, test=True)
		test_obj = self.objective(self.theta, testY, acts)
		yhat = np.argmax(acts[-1][:,:-1], 1)
		test_L = np.argmax(testY, 1)
		test_err = np.mean(test_L != yhat)

		return (train_obj, train_err, test_obj, test_err)

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
			print(self.theta.shape)
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
			print(self.mask.shape)
			del maskW
			del maskB

			return (index_X, numgradbatches, numbatches, decrease, boost, ch, f_decay)


		def run_conjgrad(ch, batchX, grad, actsbatch, precon):
			(chs, iters) = self.conjgrad(batchX, grad, ch, actsbatch, precon, self.maxiter)
			ch = chs[-1]
			iters = iters[-1]
			p = ch
			return (p, chs, ch)


		def conjgrad_backtrack(p, chs, gradbatchX, gradbatchY):
			'''
			backtrack the output of CG
			'''
			obj = self.objective(self.theta + p, gradbatchY, self.forward(self.theta+p, gradbatchX))
			for j in range(len(chs)-2, -1, -1):
				obj_chs = self.objective(self.theta + chs[j], gradbatchY, self.forward(self.theta+chs[j], gradbatchX))
				if obj < obj_chs:
					j += 1
					break
				obj = obj_chs
			p = chs[j]
			return(p, obj)


		def reduction_ratio(p, obj, obj_prev, batchX, actsbatch, grad):
			'''
			compute the reduction_ratio, with undamped quadratic approximation. 
			'''
			current_damp = self.damp
			self.damp = 0.0
			denom = self.computeGV(batchX, p, actsbatch)
			denom = 0.5 * (p * denom).sum(0)
			denom = denom - (grad * p).sum(0)
			self.damp = current_damp

			rho = (obj - obj_prev) / denom
			if obj - obj_prev > 0:
				rho = -np.inf
			return rho


		def linesearch(obj, obj_prev, grad, gradbatchX, gradbatchY):
			'''
			apply a backtracking linesearch for good update
			'''
			rate = 1.0
			c = 1e-2
			j = 0
			while j < 60:
				if obj <= obj_prev + c * rate * (grad * p).sum(0):
					break
				else:
					rate *= 0.8
					j += 1 
				obj = self.objective(self.theta, gradbatchY, self.forward(self.theta + rate * p, gradbatchX))

			if j == 60:
				rate = 0
				obj = obj_prev

			return rate


		def damping_update(rho, boost, decrease):
			'''
			using Levenberg-Marquardt to update damping parameter
			'''
			if rho < 0.25:
				self.damp *= boost
			elif rho > 0.75:
				self.damp *= decrease


		def network_update(f, rate, p):
			'''
			update the weight and bias
			'''
			self.theta += f * rate * p
			(self.W, self.b) = self.unpacknet(self.theta)


		def results_display(X, Y, testX, testY):
			print("Epoch %d:" % (self.epoch))
			(train_obj, train_err, test_obj, test_err) = self.results(X, Y, testX, testY)

			print ("\ttrain error     = %.4f" % (train_err))
			print ("\ttest error      = %.4f" % (test_err))
			print ("\tlambda          = %.8f" % (self.damp))
			print ("\tCG-decay        = %.3f" % (self.p_i))

			self.trainerr_record.append(train_err)
			self.testerr_record.append(test_err)
       


		# main
		(index_X, numgradbatches, numbatches, decrease, boost, ch, f_decay) = setup()
		count = 0
		self.trainerr_record =[]
		self.testerr_record =[]
		self.damping_record =[]

        

		for epoch in range(self.maxepoch):
			self.damping_record.append(self.damp)
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
				actsbatch = self.forward(self.theta, batchX, batchtype='GV')
				obj_prev = self.objective(self.theta, gradbatchY, acts)
				(grad, precon) = self.backward(gradbatchY, acts)
				grad = -grad
				ch = ch * self.p_i
				(p, chs, ch) = run_conjgrad(ch, batchX, grad, actsbatch, precon)
				(p, obj) = conjgrad_backtrack(p, chs, gradbatchX, gradbatchY)
				rho = reduction_ratio(p, obj, obj_prev, batchX, actsbatch, grad)
				rate = linesearch(obj, obj_prev, grad, gradbatchX, gradbatchY)
				damping_update(rho, boost, decrease)
				network_update(f_decay, rate, p)
				count += 1
			results_display(X, Y, testX, testY)
		return (self.trainerr_record, self.testerr_record, self.damping_record)


def main():
	pass

if __name__ == '__main__':
	main()









		