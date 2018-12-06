import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
from math import floor
import torchvision.transforms as transforms
from net import Net
import re
transform = transforms.Compose(transforms.ToTensor())
import sys
import amr2subgraph
import Record
from Record import record
import feature
import xlrd
import predict
from feature import treenode

learning_rate = 0.001
work_dir = './text/'
TRAIN_SBG_PATH = work_dir + 'PART-amr-release-1.0-training-proxy-subgraph.txt'
TRAIN_ANNOTATION_PATH = work_dir + 'annotation.xlsx'

def train():
	with open('./data/train_samples.pkl', 'rb') as p:
		train_set = pickle.load(p)
	with open('./intermediate/feature_template.pkl', 'rb') as p:
		featCombination = pickle.load(p)

	data, idmap = feature.process(train_set, featCombination)

	np.random.shuffle(data)
	data = torch.from_numpy(data)
	device = torch.device("cpu")

	X = data[:,:-1]
	y = data[:,-1:].type(torch.LongTensor).squeeze(1)  #CrossEntropy just receive 1-D tensor, even [32,1] need to be squeenze

	net = Net(X.size(1), 128, 2)
	net.double()
	net.to(device)

	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr = learning_rate)


	for epoch in range(500):
		running_loss = 0.0
		optimizer.zero_grad()

		y_pred = net(X)
		# print(y_pred)
		loss = loss_fn(y_pred, y)
		loss.backward()

		optimizer.step()
		running_loss = loss.item()
		print(running_loss)


	torch.save(net, './models/classifier.pth')

# def test(data):
# 	np.random.shuffle(data)
# 	data = torch.from_numpy(data)
# 	device = torch.device("cpu")
# 	X = data[:,:-1]
# 	y = data[:,-1:]
# 	net = Net(X.size(1), 128, 2)
# 	net = torch.load('./model/classifier.pth')
# 	net.to(device)
#
#
# 	precision = 0.0
# 	recall = 0.0
# 	TP = 0.0
# 	FP = 0.0
# 	TN = 0.0
# 	FN = 0.0
# 	with torch.no_grad():
# 		outputs = net(X)
# 		# _, predicted = torch.max(outputs, 1)
# 		prediction = torch.max(F.softmax(outputs),1)[1]
# 		total = y_test.size(0)
#
# 		for i in range(len(prediction)):
# 			if prediction[i] == 1 and y_test[i] == 1:
# 				TP += 1
# 			if prediction[i] == 0 and y_test[i] == 0:
# 				TN += 1
# 			if prediction[i] == 1 and y_test[i] == 0:
# 				FP += 1
# 			if prediction[i] == 0 and y_test[i] == 1:
# 				FN += 1
# 		print(TP, FP, TN, FN)
# 		precision += TP / (TP + FP)
# 		recall += TP / (TP + FN)
# 		Fmeasure = 2 / (1 / precision + 1 / recall)
#
# 	print('precision:%f %%' % precision)
# 	print('Recall:%f %%' % recall)
# 	print('F-measure: %f' % Fmeasure)
# 	print('total %f' % len(prediction))


def main(filename):
	# f = open('./text/'+filename+'.all.basic-abt-brown-verb.parsed','r', encoding='utf8')
	# g = open('./text/'+filename+'.sbg','w')
	# raw_amrs = re.split("\n\n", f.read().strip())

	#prepare for amr2sbg
	# subgraph = amr2subgraph.process(raw_amrs)

	#prepare for record
	# sbg_list = subgraph.split('-' * 50 + '\n')[1:]
	# with open(TRAIN_SBG_PATH,'r',encoding = 'utf8') as f:
	# 	train_sbg = f.read().split('-' * 50 + '\n')[2:]
	# train_rec = combine(train_sbg, TRAIN_ANNOTATION_PATH)
    #
	# with open('./data/first_data.pkl','rb') as p:
	# 	first_record = pickle.load(p)
	# with open('./data/second_data.pkl','rb') as p:
	# 	second_record = pickle.load(p)
	# train_rec = first_record + second_record
	# file_len = len(train_rec)
	# random.shuffle(train_rec)
    #
	# l = floor(0.8 * file_len)
	# train_set = train_rec[:l]
	# test_set = train_rec[l:]

	train()
def test():
	with open('./data/test_samples.pkl', 'rb') as p:
		test_set = pickle.load(p)

	with open('./intermediate/feature_template.pkl', 'rb') as p:
		featCombination = pickle.load(p)

	# train_data, idmap = feature.process(train_set, featCombination)
	# train(train_data)

	test_data, test_idmap = feature.process(test_set, featCombination)
	binders = predict.predict(test_data)
	data, y_pred = zip(*binders)
	data = list(data)
	y_pred = list(y_pred)
	print(y_pred)
	records = [test_idmap[str(int(d[0]))] for d in data]
	evaluate(y_pred, records)


def evaluate(y_pred, records):
	TP = 0
	TP_list = []
	TN = 0
	TN_list = []
	FP = 0
	FP_list = []
	FN = 0
	FN_list = []

	for i in range(len(y_pred)):
		if y_pred[1] == 1 and records[i].annotation == 1:
			TP += 1
			TP_list.append(records[i].graph)
		elif y_pred[1] == 1 and records[i].annotation == 0:
			FP +=  1
			FP_list.append(records[i].graph)
		elif y_pred[i] == 0 and records[i].annotation == 1:
			FN += 1
			FN_list.append(records[i].graph)
		elif y_pred[i] == 0 and records[i].annotation == 0:
			TN += 1
			TN_list.append(records[i].graph)
	total = len(y_pred)
	print(TP,TN,FP,FN)
	precision = TP / (TP + FN)
	recall = TP / (TP + FP)
	f_measure = 2 / (1.0 / precision + 1.0 / recall)
	print('precision: %f\nrecall:%f\nf_measure:%f' %(precision,recall,f_measure))
	with open('test_analysis.txt','w', encoding='utf8') as f:
		f.write('\n'.join(TP_list))
		f.write('#'*50)
		f.write('\n'.join(FP_list))
		f.write('#' * 50)
		f.write('\n'.join(FN_list))
		f.write('#' * 50)
		f.write('\n'.join(TN_list))








	# with open('./text/'+filename+'.sbg', 'w',encoding = 'utf8') as f:
	# 	f.write(subgraph)



	# records = Record.ProduceRecords(sbg_list)
	# print("Processing %d records in training data" % len(records))
	#we need to save record here for future's search

	#prepare for extract feature

def combine(subgraph, annotationPath):
	'''
	This function is used for training data specifically
	'''
	lino = 0
	annotation = xlrd.open_workbook(annotationPath).sheets()[0]
	maxrows = annotation.nrows
	records = []
	for s in subgraph:
		line = s.split('\n')
		index = re.search("id\s(\S+)\.(\d+)",line[0])	#/d can only match one digit
		aid = index.group(1)
		sid = index.group(2)
		raw_text = ""
		for i in range(len(line)):
			if line[i].startswith('#'):
				if line[i].startswith('# ::snt'):
					raw_text = line[i][7:]
				else:
					continue
			else:
				break
		sg = '\n'.join(line[i:])
		if not sg:
			continue
		sg = sg.split('\n\n')[:-1]


		for e in sg:
			r = re.search("\((\d+)\)",e)
			if not r:
				print('sb')
			ind = int(r.group(1)) 	# a integer
			graph = e[r.end(1)+1:]
			try:
				while lino < maxrows and annotation.row(lino)[1].value != aid+'.'+sid:
					#print(annotation.row(lino)[1].value)
					lino += 1

				i = 1
				while lino + i < maxrows and ind != annotation.row(lino+i)[2].value:
					#print(annotation.row(lino+i)[2].value)
					i += 1
				if annotation.row(lino+i)[3].value == 'y' or annotation.row(lino+i)[3].value == 'yes':
					y = 1.0
				else:
					y = 0.0
			except IndexError:
				print(aid+'.'+sid)
			records.append(record(raw_text,graph,aid,sid,y))
	return records

def error_analysis():
	'''
	split the annotated data as 4:1, and fix the training and testing data.
	'''



if __name__ == '__main__':
	filename = sys.argv[1]
	filename = filename.split('/')[-1]
	main(filename)