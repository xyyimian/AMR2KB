# -*- coding:utf-8 -*-
import pickle
from sklearn.decomposition import PCA
import sys
sys.path.append('.')
from feature_extract.tuple import subGraph
import numpy as np
np.set_printoptions(threshold=np.inf)
from net import Net
import torch
import torch.nn.functional as F



def predict(filename):
	# with open ('./classifier.pkl','rb') as p:					#!
	# 	clf = pickle.load(p)
	# with open('../feature_extract/'+filename.split('/')[-1]+'.ft.pkl','rb') as p:
	# # with open('../feature_extract/traindata.pkl','rb') as p:
	# 	data = pickle.load(p)
	# with open('../feature_extract/'+filename.split('/')[-1]+'.tp.pkl','rb') as p:
	# 	subGraphs = pickle.load(p)
	# with open('./pca.pkl','rb') as p:
	# 	pca = pickle.load(p)

	
	# X = data[:, :6]
	# # X = pca.transform(X)
	# pred = clf.predict(X)

	# results = open('./results.txt','w')
	# print(len(pred))
	# print(pred)

	# # assert len(subGraphs) == len(pred)
	# # for i in range(len(subGraphs)):
	# # 	if pred[i] == 1:
	# # 		results.write(subGraphs[i].graph)
	# # 		results.write('\n\n')
	# results.close()

	with open('./feature_extract/'+filename.split('/')[-1]+'.ft.pkl','rb') as p:
	# with open('../feature_extract/traindata.pkl','rb') as p:
		data = pickle.load(p)
		data = data[:,:-1]
		data = torch.from_numpy(data)

	with open('./feature_extract/'+filename.split('/')[-1]+'.tp.pkl','rb') as p:
		subGraphs = pickle.load(p)

	# print(data.size(1))
	net = Net(data.size(1), 128, 128, 2)
	net = torch.load('./train/classifier.pkl')
	net.eval()

	prediction = torch.max(F.softmax(net(data)),1)[1]
	print(prediction)
	results = open('./results.txt','w')

	assert len(subGraphs) == len(prediction)
	for _, e in enumerate(subGraphs):
		if prediction[_] == 1:
			results.write(e.graph)
			results.write('\n\n')
	results.close()




if __name__ == '__main__':
	filename = sys.argv[1]
	predict(filename)


