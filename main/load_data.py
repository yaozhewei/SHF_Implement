import numpy as np

def load_data():

	# load data
	data_path = '../data/'
	data = []
	data.append(np.loadtxt(data_path + 'usps0', delimiter=','))
	data.append(np.loadtxt(data_path + 'usps1', delimiter=','))
	data.append(np.loadtxt(data_path + 'usps2', delimiter=','))
	data.append(np.loadtxt(data_path + 'usps3', delimiter=','))
	data.append(np.loadtxt(data_path + 'usps4', delimiter=','))
	data.append(np.loadtxt(data_path + 'usps5', delimiter=','))
	data.append(np.loadtxt(data_path + 'usps6', delimiter=','))
	data.append(np.loadtxt(data_path + 'usps7', delimiter=','))
	data.append(np.loadtxt(data_path + 'usps8', delimiter=','))
	data.append(np.loadtxt(data_path + 'usps9', delimiter=','))

    # load permutation matrix
	perms = np.loadtxt(data_path + 'permutation', delimiter=',') - 1
	perms = perms.astype(int)

    # set train/ test data, 
	train = np.zeros((8000, 256))
	test = np.zeros((3000, 256))
	train_labels = np.zeros((8000, 10))
	test_labels = np.zeros((3000, 10))
	e = np.eye(10)
	for i in range(10):
		train[i * 800 : (i+1) * 800, :] = data[i][:,perms[:800]].T
		test[i * 300 : (i+1) * 300, :] = data[i][:,perms[800:]].T
		train_labels[i * 800 : (i+1) * 800, :] = np.tile(e[i,:], (800, 1))
		test_labels[i * 300 : (i+1) * 300, :] = np.tile(e[i,:], (300, 1))  

    # rescale the input
	train = train / 255.0
	test = test / 255.0
	return (train, test, train_labels, test_labels)   

def normization_data(train, test):
	'''
    normize the data
    '''
	eps = 0.01 
	train_mean = np.mean(train, 0)
	train_std = np.sqrt(np.var(train, 0) + eps)
	trainX = (train - train_mean) / train_std
	testX = (test - train_mean) / train_std
	return (trainX, testX)


def main():
    pass

if __name__ == '__main__':
    main()   	