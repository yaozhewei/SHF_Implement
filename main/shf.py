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
				 damp = 0.1):

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

	def init_parameters(self, X):
		'''
		initialize the parameters: weights and bias.
		'''
		(m,n) = np.shape(X)
		#print(m,n)
		(W,b) = ([0] * len(self.layers), [0] * len(self.layers))

		if self.activations[0] == 'ReLU':
			b_ReLU_shift = 0.1
		else:
			b_ReLU_shift = 0

		r = np.sqrt(6) / np.sqrt(n + self.layers[0] + 1)
		W[0] = 2 * r * np.random.rand(n, self.layers[0]) - r
		b[0] = np.zeros((1, self.layers[0])) + b_ReLU_shift

		for i in range(1, len(self.layers)):
			if self.activations[i] == 'ReLU':
				b_ReLU_shift = 0.1
			else:
				b_ReLU_shift = 0

			r = np.sqrt(6) / np.sqrt(n + self.layers[i-1] + self.layers[i] + 1)
			W[i] = 2 * r * np.random.rand(self.layers[i-1], self.layers[i]) - r
			b[i] = np.zeros((1, self.layers[i])) + b_ReLU_shift
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
		return theta


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

		for i in range(len(self.layers) - 1):
			num_w = self.layers[i] * self.layers[i+1]
			num_b = self.layers[i+1]
			W[i+1] = np.reshape(theta[index : index + num_w], (self.layers[i], self.layers[i+1]), order = 'F')
			index += num_w
			b[i+1] = np.reshape(theta[index : index + num_b], (1, self.layers[i+1]), order = 'F')
			index += num_b
		
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
			# forget to multiply (1- drop) 
			if test and i == 0:
				preacts = np.dot(acts[i], np.concatenate((W[i] * (1 - self.dropout[0]), b[i])))
			elif test and i > 0:
				preacts = np.dot(acts[i], np.concatenate((W[i] * (1 - self.dropout[1]), b[i])))
			else:
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

		# change the 1e-16 to 1e-20
		if objfunction == 'MSE':
			out = 0.5 * np.sum((acts[-1][:,:-1] - Y)**2) / batchsize
		elif objfunction == 'cross-entropy':
			out = -np.sum(Y * np.log(acts[-1][:,:-1] + 1e-20) + (1 - Y) * np.log(1 - acts[-1][:,:-1] + 1e-20)) / batchsize
		elif objfunction == 'softmax-entropy':
			out = -np.sum(Y * np.log(acts[-1][:,:-1] + 1e-20)) / batchsize

		# add penalty
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
		grad += self.weight_cost * (self.mask * self.theta)
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

		RLx = R[:,:-1] / batchsize

		for i in range(len(self.layers) - 1, -1, -1):
			delta = np.dot(acts[i].T, RLx)
			dVW[i] = delta[:-1,:]
			dVb[i] = delta[-1,:]

			if i > 0:
				if self.activations[i-1] == 'linear':
					RLx = np.dot(RLx, np.concatenate((self.W[i], self.b[i])).T)
				elif self.activations[i-1] == 'logistic':
					RLx = np.dot(RLx, np.concatenate((self.W[i], self.b[i])).T) * (acts[i] * (1 - acts[i]))
				elif self.activations[i-1] == 'ReLU':
					RLx = np.dot(RLx, np.concatenate((self.W[i], self.b[i])).T) * (acts[i] > 0)
			RLx = RLx[:,:-1]

		Jv = self.packnet(dVW, dVb)
		Jv += self.weight_cost * (self.mask * v)
		Jv += self.damp * v

		return Jv

	def computeJv(self, X, v, acts=None):
		'''
		compute gauss-newton vector: JV
		'''
		Jv = self.R_backward(acts, self.R_forward(acts, v), v)

		return Jv

	def conjgrad(self, X, b, x0, acts, precon, maxiter):
		'''
		CG for solving Gx = -grad.
		'''
		XS = [0] * maxiter
		r = self.computeJv(X, x0, acts) - b
		y = r / precon
		p = -y
		x = x0
		delta_new = (r * y).sum()

		for i in range(maxiter):

			Ap = self.computeJv(X, p, acts)
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

			XS[i] = x

		return XS

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
			self.init_parameters(X)
			index_X = list(range(self.numdata))
			np.random.shuffle(index_X)
			numgradbatches = len(index_X) // self.gradbatchsize
			numbatches = self.gradbatchsize // self.batchsize

			self.theta = self.packnet(self.W, self.b)

			numparameters = len(self.theta)
			self.mask = np.ones((numparameters, 1))
			(maskW, maskB) = self.unpacknet(self.mask)
			for i in range(len(self.layers)):
				maskB[i][:] = 0
			self.mask = self.packnet(maskW, maskB)
			decrease = 0.99
			boost = 1 / decrease
			cg_ini = np.zeros((numparameters, 1))
			step_decay = 1

			del maskW
			del maskB

			return (index_X, numgradbatches, numbatches, decrease, boost, cg_ini, step_decay)


		def run_conjgrad(cg_ini, batchX, grad, actsbatch, precon):
			cg_all = self.conjgrad(batchX, grad, cg_ini, actsbatch, precon, self.maxiter)
			cg_ini = cg_all[-1]
			p = cg_ini
			return (p, cg_all, cg_ini)


		def conjgrad_backtrack(p, cg_all, gradbatchX, gradbatchY):
			'''
			backtrack the output of CG
			'''
			obj = self.objective(self.theta + p, gradbatchY, self.forward(self.theta+p, gradbatchX))
			for j in range(len(cg_all)-2, -1, -1):
				obj_cgall = self.objective(self.theta + cg_all[j], gradbatchY, self.forward(self.theta+cg_all[j], gradbatchX))
				if obj < obj_cgall:
					j += 1
					break
				obj = obj_cgall
			p = cg_all[j]
			return(p, obj)


		def reduction_ratio(p, obj, obj_prev, batchX, actsbatch, grad):
			'''
			compute the reduction_ratio, with undamped quadratic approximation. 
			'''
			current_damp = self.damp
			self.damp = 0.0
			denom = self.computeJv(batchX, p, actsbatch)
			denom = 0.5 * (p * denom).sum(0)
			denom = denom - (grad * p).sum(0)
			self.damp = current_damp

			rho = (obj - obj_prev) / denom
			if obj - obj_prev > 0:
				rho = -np.inf

			#print(rho)
			return rho


		def linesearch(obj, obj_prev, grad, gradbatchX, gradbatchY):
			'''
			apply a backtracking linesearch for good update
			'''
			learning_rate = 1.0
			c = 1e-2
			j = 0
			while j < 60:
				if obj <= obj_prev + c * learning_rate * (grad * p).sum(0):
					break
				else:
					learning_rate *= 0.8
					j += 1 
				obj = self.objective(self.theta, gradbatchY, self.forward(self.theta + learning_rate * p, gradbatchX))

			if j == 60:
				learning_rate = 0
				obj = obj_prev

			return learning_rate


		def damping_update(rho, boost, decrease):
			'''
			using Levenberg-Marquardt to update damping parameter
			'''
			if rho < 0.25 or np.isnan(rho):
				self.damp *= boost
			elif rho > 0.75:
				self.damp *= decrease
			#print(self.damp)


		def network_update(f, learning_rate, p):
			'''
			update the weight and bias
			'''
			self.theta += f * learning_rate * p
			(self.W, self.b) = self.unpacknet(self.theta)


		def results_display(X, Y, testX, testY):
			print("Epoch %d:" % (self.epoch))
			(train_obj, train_err, test_obj, test_err) = self.results(X, Y, testX, testY)

			print ("\ttrain error     = %.6f" % (train_err))
			print ("\ttest error      = %.6f" % (test_err))
			print ("\tlambda          = %.6f" % (self.damp))

			self.trainerr_record.append(train_err)
			self.testerr_record.append(test_err)
       


		# main
		(index_X, numgradbatches, numbatches, decrease, boost, cg_ini, step_decay) = setup()

		count = 0
		
		self.trainerr_record =[]
		self.testerr_record =[]
		self.damping_record =[]

		for epoch in range(self.maxepoch):
			self.damping_record.append(self.damp)
			if epoch > 0:
				step_decay *= 0.998
				#step_decay = 1
				if self.cgdecay_ini < self.cgdecay_fnl:
					self.cgdecay_ini = np.minimum(1.01 * self.cgdecay_ini, self.cgdecay_fnl)
					if self.cgdecay_ini > self.cgdecay_fnl:
						self.cgdecay_ini = self.cgdecay_fnl

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
				cg_ini = cg_ini * self.cgdecay_ini
				(p, cg_all, cg_ini) = run_conjgrad(cg_ini, batchX, grad, actsbatch, precon)
				#p_med = p
				(p, obj) = conjgrad_backtrack(p, cg_all, gradbatchX, gradbatchY)
				#p=p_med
				rho = reduction_ratio(p, obj, obj_prev, batchX, actsbatch, grad)
				#print(rho)
				learning_rate = linesearch(obj, obj_prev, grad, gradbatchX, gradbatchY)
				damping_update(rho, boost, decrease)
				network_update(step_decay, learning_rate, p)
				count += 1
			results_display(X, Y, testX, testY)
		return (self.trainerr_record, self.testerr_record, self.damping_record)


def main():
	pass

if __name__ == '__main__':
	main()









		