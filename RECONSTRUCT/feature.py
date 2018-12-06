# -*- coding:utf-8
from Record import record
from constants import *
import pickle
import sys
sys.path.append('.')											#!
import amr2subgraph
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tokenize import word_tokenize
from functools import partial
from functools import lru_cache
import re
import torch
# import constant


lemmatizer = lru_cache(maxsize=50000)(partial(WordNetLemmatizer().lemmatize,pos = 'v'))

class treenode:
	def __init__(self, value = ""):
		self.next_edges = [] 		#label
		self.next_nodes = []		#reference
		self.value = value


hashTable = set()
feature_funcs = []
def register(feature_func):
	print('extracting feature: ',feature_func.__name__)
	feature_funcs.append(feature_func)
	return feature_func

@register
def pred2f(data, root, featCombination):
	'''
	predicate class as feature
	'''
	pred_list = featCombination[2]
	try:
		idx = list(pred_list).index(root.ful_name.split('-')[0])
	except ValueError:
		idx = -1
	data.PredClass = idx

@register
def Verb2f(data, root, featCombination):
	TempModNum = 0
	SayTermNum = 0
	WishTermNum = 0
	tokens = word_tokenize(data.raw_text)
	for e in nltk.pos_tag(tokens):
		if 'VB' in e[1]:
			initiative =  lemmatizer(e[0])
			if initiative in FutureHaving:
				TempModNum += 1
			if initiative in SayTerm:
				SayTermNum += 1
			if initiative in Wish_61:
				WishTermNum += 1


	data.TempMod = TempModNum
	data.SayTerm = SayTermNum
	data.WishTerm = WishTermNum
#
# @register
# def Say2f(data,root, featCombination):
#
# 	for w in data.raw_text.split():
# 		# print(lemmatizer(w))
# 		if lemmatizer(w) in SayTerm:
# 			SayTermNum += 1
#
#
# @register
# def Wish2f(data, root, featCombination):
#
# 	for w in data.raw_text.split():
# 		if lemmatizer(w) in Wish_61:
# 			WishTermNum += 1


@register
def ne2f(data, root, featCombination):
	'''
	compute the mark according to number and level of ne
	'''
	neMark = 0
	root.ancillary = 0
	DFS(root, [neMark],1)
	data.neMark = neMark

def DFS(node, results, level):
	if node.ancillary == 0:
		node.ancillary = 1
		if node.is_entity:
			results[0] += 1 / level
		for child in node.next_nodes:
			child.ancillary = 0
			DFS(child, results, level+1)



@register
def struc2f(data, root, featCombination):
	feature_list = featCombination[1]
	struc = [0] * (len(feature_list) + 1)

	for _, f in enumerate(feature_list):
		isExist = True
		edgeFeat = f.next_edges
		realFeat = [e.edge_label for e in root.next_nodes]
		for e in edgeFeat:
			if e not in realFeat:
				isExist = False
				break
			if f.next_nodes:
				secondLevelLabel = f.next_nodes[0].next_edges
				idx = realFeat.index(f.next_edges[0])
				for l2 in secondLevelLabel:
					if l2 not in [e.edge_label for e in root.next_nodes[idx].next_nodes]:
						isExist = False
						break
		if isExist:
			struc[_] = 1
		if root.ful_name == f.value:
			struc[-1] = 1
	data.struc = struc


@register
def labelNum2f(data, root, featCombination):
	'''
	calculate the number of different label in first two level
	'''
	labelList = featCombination[0]
	labelNum = [0]*len(labelList)
	lst1 = [e.edge_label for e in root.next_nodes]
	for i,id1 in enumerate(lst1):
		try:
			idx = labelList.index(id1)
		except ValueError:
			idx = 0
		labelNum[idx] += 1
		lst2 = [e.edge_label for e in root.next_nodes[i].next_nodes]
		for j in lst2:
			try:
				idx = labelList.index(j)
			except ValueError:
				idx = 0
			labelNum[idx] += 1
	data.labelNum = labelNum

def myHash(treeroot):
	s = ""
	for i, e1 in enumerate(sorted(treeroot.next_edges)):
		s = s + str(i) + e1
		if not treeroot.next_nodes:
			continue
		for j, e2 in enumerate(sorted(treeroot.next_nodes[i].next_edges)):
			s = s + str(j) + e2
	return s


def find_feature(root_list):
	'''
	input: pos training data's root_list
	otuput: feature_list
	'''
	feature_list = []
	labelList = set()

	for root in root_list:
		labelList.add(root.edge_label)
		if not root.next_nodes:
			r = treenode(root.ful_name)
			feature_list.append(r)
			return
		#Firstly generate feature with pred and two of its children
		FirstLevelTuple = []
		for c1 in root.next_nodes:
			for c2 in root.next_nodes:
				if c1 is c2:
					continue
				if (c2.edge_label, c1.edge_label) not in FirstLevelTuple:
					FirstLevelTuple.append((c1.edge_label, c2.edge_label))

		FirstLevelStruc = []

		for _, flt in enumerate(FirstLevelTuple):
			r = treenode(root.ful_name)
			r.next_edges += [flt[0], flt[1],]
			h = myHash(r)
			if h  not in hashTable:
				hashTable.add(h)
				FirstLevelStruc.append(r)

		#Secondly, generate feature with path from pred to its leaf
		PathStruc = []
		for c1 in root.next_nodes:
			if not c1.next_nodes:
				labelList.add(c1.edge_label)
				r = treenode(root.ful_name)
				# n1 = treenode()
				# r.next_nodes.append(n1)
				r.next_edges.append(c1.edge_label)
				h = myHash(r)
				if h not in hashTable:
					hashTable.add(h)
					PathStruc.append(r)
				continue
			for c2 in c1.next_nodes:
				labelList.add(c2.edge_label)
				n1 = treenode()
				n1.next_edges.append(c2.edge_label)
				r = treenode(root.ful_name)
				r.next_nodes.append(n1)
				r.next_edges.append(c1.edge_label)
				h = myHash(r)
				if h not in hashTable:
					hashTable.add(h)
					PathStruc.append(r)
		
		feature_list = feature_list + FirstLevelStruc + PathStruc
	return labelList, feature_list


def feature_template_extract(train_rec):
	feature_list = []
	pred_list = set()
	root_list = []

	counter = 0
	for s in train_rec:
		amr = s.graph
		amr_nodes_acronym, root, _ = amr2subgraph.amr_reader(amr)
		pred_list.add(root.ful_name.split('-')[0])

		if s.annotation == 1.0:
			counter += 1
			root_list.append(root)
	labelList, feature_list = find_feature(root_list)
	print('The pos sample num in training data:%d' %counter)
	return (labelList, feature_list, pred_list)

def extract_feature(feature_template):
	pass

def process(records, featCombination):
	'''
	input: records 
	output: array-like training data
	'''
	idmap = {}
	for i,s in enumerate(records):
		s.id = len(idmap)
		idmap[str(s.id)] = s
		_, root, _ = amr2subgraph.amr_reader(s.graph)
		for f in feature_funcs:
			f(s, root, featCombination)
	
	attrs = [attr for attr in dir(records[0]) if attr not in dir(records)]

	for a in ['graph','annotation','labelNum','raw_text','struc']:
		attrs.remove(a)
	print(attrs, 'labelNum struc')
	traindata = []
	attrs = ['id','PredClass', 'SayTerm', 'TempMod', 'WishTerm', 'neMark']
	for e in records:
		lst = [getattr(e,a) for a in attrs] + e.labelNum + e.struc
		traindata.append(lst)
	
	traindata = np.array(traindata,dtype = 'float64')
	return traindata, idmap
	
def m_train(train_rec):
	labelList, pred_list, feature_list = feature_template_extract(train_rec)
	labelList = ['MISC']+list(labelList)
	featCombination = (labelList, pred_list, feature_list)
	with open('feature_template.pkl', 'wb') as p:
		pickle.dump(featCombination, p)
	return featCombination


if __name__ == '__main__':
	mode = sys.argv[1]
	filename = sys.argv[2]
	main(mode, filename)