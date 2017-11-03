import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import itertools as it


class Mrama:
	'''The class Mrama trains the learning automata based multi-class classifier with
		the training data set.'''
	def __init__(self, *args,**kwargs):
		self.params = kwargs
		self.theta = 0

	def fit(self, x, y):
		x = np.tile(x, (100,1))
		y = y * 100
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
					self.theta[inneridx] = self.theta[inneridx] + self.params['l_rate'] * Beta * np.insert(val, 0, 1) * (1 - genFunc[inneridx]) * np.ones((1, x.shape[1]+1)) + self.params['l_rate'] * self.params['K']  * (h[inneridx] - self.theta[inneridx]) 
				else:
					self.theta[inneridx] = self.theta[inneridx] - self.params['l_rate'] * Beta * np.insert(val, 0, 1) * genFunc[inneridx] * np.ones((1, x.shape[1]+1)) + self.params['l_rate'] *self.params['K']* (h[inneridx] - self.theta[inneridx])	
			h = self.theta					
			for idxh in np.nditer(h, op_flags=['readwrite']):
				if idxh[...] > self.params['L'] :
					idxh[...] = self.params['L']
				elif idxh[...] < -self.params['L'] :
					idxh[...] = -self.params['L']
				else:
					pass
		return self.theta
	def cross_validation(self, x, y, cv):
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1.0/cv)
		cv_theta = self.fit(x_train, y_train)
		cv_predict = self.predict(x_test)
		cv_accuracy = 0.0
		for i,j in it.izip(y_test, cv_predict):
			if i == j:
				cv_accuracy += 1
		return cv_accuracy*100/len(y_test)


	def predict(self, x):
		y_pred = list()
		for idx, val in enumerate(x):
			z = np.exp(np.matmul(np.insert(val, 0, 1), np.transpose(self.theta)))
			genFunc = [i/np.sum(z) for _, i in enumerate(z)]
			y_pred.append(np.argmax(genFunc))
		return y_pred



		















# def mrama():
# 	print "Hi"


# def main():
# 	pd.read_table("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
# 	pd.read_table("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data")
	

# if __name__=="__main__":
# 	c = MramaTrain()
# 	print MramaTrain.__doc__
# 	# main()