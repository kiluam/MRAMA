'''The license belongs to s maulik the author of this code. For publishing consent of
the author is necessary.'''

import pandas as pd
import mrama as mr
import itertools as it
from sklearn.model_selection import train_test_split




def main():
	''' Evaluates the performance of MRAMA on Iris data set. To find the optimal leaarning parameters 
	    uncomment line 27 and 28 and comment line 26, 29-35. For evaluating the performance of MRAMA with
	    under best possible combination of paramters, comment line 27 and 28 and uncomment line 26, 29-35.'''
	attributeVar = ['sLength', 'sWidth', 'pLength', 'pWidth' , 'flowerType']
	df = pd.read_table("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", delimiter = ',', names = attributeVar) #For reading from Url
	# df = pd.read_table("iris.data.txt", delimiter = ',', names = attributeVar) #For reading from local disk
	y = df['flowerType'].astype('category').cat.codes
	y = y.tolist()
	df.drop('flowerType', axis = 1, inplace = True)
	for columns in df:
		df[columns] = df[columns]/df[columns].max()
	x = df.as_matrix()
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
	clf = mr.Mrama( 0.1, 15, 100, 200)
	# scores = mr.Mrama.cross_validation(x_train, y_train, 5, 10, niters = [50, 100, 200, 500],l_rate = [0.1, 0.2, 0.3], K = [15, 100], L = [100, 1000])
	# print scores #cross-validation accuracy in %
	model = clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test, model)
	test_accuracy = 0
	for i,j in it.izip(y_test, y_pred):
		if i == j:
			test_accuracy += 1
	print test_accuracy #Absolute Accuracy
	
if __name__=="__main__":
	main()
