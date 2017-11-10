import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import itertools as it
import numpy as np


class Mrama:
	'''The class MramaTrain trains the learning automata based multi-class classifier with
		the training data set. The inputs are the learning rate, the magnitude of penalty
		and the window width'''
	def __init__(self, l_rate, K, L, niters):
		self.l_rate = l_rate
		self.K = K
		self.L = L
		self.niters = niters
		self.theta = 0

	def fit(self, x, y):
		'''The training method of the class Mrama. The mandatory inputs are 
		the attributes in form of numpy array and the class as list. The method returns
		the states of the automata as learned in the due process of training'''
		x = np.tile(x, (self.niters,1))
		y = y * self.niters
		self.theta = np.random.ranf((len(set(y)), x.shape[1]+1))
		h = self.theta
		for idx, val in enumerate(x):
			z = np.exp(np.matmul(np.insert(val, 0, 1), np.transpose(h)))
			# print z.shape
			genFunc = [i/np.sum(z) for _, i in enumerate(z)]
			if np.argmax(genFunc) == y[idx]:
				Beta = 1
			else:
				Beta = -1
			for inneridx, innerval in enumerate(self.theta):
				if inneridx == np.argmax(genFunc):
					self.theta[inneridx] = self.theta[inneridx] + self.l_rate * Beta * np.insert(val, 0, 1) * (1 - genFunc[inneridx]) * np.ones((1, x.shape[1]+1)) + self.l_rate * self.K  * (h[inneridx] - self.theta[inneridx]) 
				else:
					self.theta[inneridx] = self.theta[inneridx] - self.l_rate * Beta * np.insert(val, 0, 1) * genFunc[inneridx] * np.ones((1, x.shape[1]+1)) + self.l_rate *self.K* (h[inneridx] - self.theta[inneridx])	
			h = self.theta					
			for idxh in np.nditer(h, op_flags=['readwrite']):
				if idxh[...] > self.L :
					idxh[...] = self.L
				elif idxh[...] < -self.L :
					idxh[...] = -self.L
				else:
					pass
		return self.theta

	@classmethod
	def _parameter(cls, l_rate, K, L, niters):
		'''This is a private method'''
		return cls(l_rate, K, L, niters)

	@classmethod
	def cross_validation(self, x, y, kfold = 1, n_kfold = 1, **kwargs):	
		'''This function is equivalent to traditional gridsearchcv in sklearn. The 
		inputs are the attributes and the class of the training set, kfold and 
		repeatedkfold, learning rate, penalty value, window width, number of repetitions
		of the training data. The function will display the cross-validation accuracy. Example of calling the method 
		cross_validation(x_train, y_train, 5, 10, niters = [50, 100, 200, 500],l_rate = [0.1, 0.2, 0.3], K = [15, 100], L = [100, 1000])'''
		for rate in kwargs['l_rate']:
			for k in kwargs['K']:
				for l in kwargs['L']:
					for iters in kwargs['niters']:
						print "Learning rate = {}, K = {}, L = {}, Repetitions = {}". format(rate, k, l, iters)
						cv_accuracy = list()
						for n_fold in xrange(1, n_kfold+1):						
							cv = self._parameter(rate, k, l, iters)
							x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1.0/kfold)
							cv_theta = cv.fit(x_train, y_train)
							cv_predict = cv.predict(x_test)
							cv_var = 0.0
							for i,j in it.izip(y_test, cv_predict):
								if i == j:
									cv_var += 1						
							cv_accuracy.append(cv_var*100/len(y_test))
						print "CV Accuracy in % = {}". format(np.mean(cv_accuracy))
						print "-----****-----"
		return 0 #cv_accuracy*100/len(y_test)


	def predict(self, x):
		'''Prediction method. The input is the attributes and the method returns the
		predicted class of the trained classifier'''
		y_pred = list()
		for idx, val in enumerate(x):
			z = np.exp(np.matmul(np.insert(val, 0, 1), np.transpose(self.theta)))
			genFunc = [i/np.sum(z) for _, i in enumerate(z)]
			y_pred.append(np.argmax(genFunc))
		return y_pred





		













