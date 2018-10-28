# -*- coding:utf
from sklearn import svm
from sklearn.decomposition import PCA
from numpy import ravel, random

import numpy as np
np.set_printoptions(threshold=np.inf)
import random as rd
from math import *
import array
import sys
import pickle
sys.path.append('../feature_extract')
from tuple import subGraph
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

def train():
	with open('../feature_extract/traindata.pkl','rb') as p:
		data = pickle.load(p)
# 25 / 31?


	X = data[:, :-1]
	
	# pca = PCA(n_components = 2, svd_solver='randomized')
	# X = pca.fit_transform(X)
	# with open('./pca.pkl','wb') as p:
	# 	pickle.dump(pca,p)
	y = ravel(data[:,-1:])

	# y = ravel(np.array([1.0]*len(X)))
	# for i in range(len(y)):
	# 	index = rd.randint(0,900)
	# 	y[index] = 0.0

	

	# param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
	param_grid = [{'C': [1,10,100,1000], 'gamma': [0.001,0.0001], 'kernel': ['rbf']}]
	svc = svm.SVC()
	clf = GridSearchCV(svc, param_grid, cv=5)
	clf.fit(X,y)
	print(clf.get_params())

	random.shuffle(X)

	l = len(X)
	s = floor(0.8*l)
    
	X_train = X[:s]
	X_test = X[s:]
	y_train = y[:s]
	y_test = y[s:]
    
	p = clf.predict(X_test)
	print(p)

	print('-'*30)
	print(len(X))
    
	TP = sum([int(p[i] == y_test[i]) for i in range(len(y_test)) if p[i] == 1])
	FP = sum(p) - TP
	TN = sum([int(p[i] == y_test[i]) for i in range(len(y_test)) if p[i] == 0])
	FN = len(p) - sum(p) - TN
	P = TP / (TP + FP)
	R = TP / (TP + FN)
	F = 2*P*R/(P+R)
    
	print('Precision:',P)
	print('Recall:', R)
	print('F-measure:', F)
    
    
	with open('./classifier.pkl','wb') as p:
		pickle.dump(clf, p)
	# k_fold = KFold(n_splits = 10)

	
	# score = cross_val_score(clf, X, y, cv = 10)
	# print('rbf:\n', score)	


if __name__ == '__main__':
	train()