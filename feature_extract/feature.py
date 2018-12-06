# -*- coding:utf-8
from tuple import subGraph
from constants import *
import pickle
import sys
sys.path.append('.')											#!
from subgraph import amr_reader
import numpy as np
from nltk.stem import WordNetLemmatizer
from functools import partial
import torch
# import constant


lemmatizer = partial(WordNetLemmatizer().lemmatize,pos = 'v')

class treenode:
	def __init__(self, value = ""):
		self.next_edges = [] 		#label
		self.next_nodes = []		#reference
		self.value = value



feature_list = []
pred_list = set()
labelList = set()
hashTable = set()
feature_funcs = []
def register(feature_func):
	print('extracting feature: ',feature_func.__name__)
	self.feature_funcs.append(feature_func)
	return feature_func

@register
def pred2f(data, root):
	'''
	predicate class as feature
	'''
	try:
		idx = list(self.pred_list).index(root.ful_name)
	except ValueError:
		idx = -1
	data.PredClass = idx


@register
def TempMod2f(data, root):
	TempModNum = 0
	for w in data.raw_text:
		if lemmatizer(w) in FutureHaving:
			TempModNum += 1
	data.TempMod = TempModNum

@register
def Say2f(data,root):
	SayTermNum = 0
	for w in data.raw_text.split():
		# print(lemmatizer(w))
		if lemmatizer(w) in SayTerm:
			SayTermNum += 1
	data.SayTerm = SayTermNum

@register
def Wish2f(data, root):
	WishTermNum = 0
	for w in data.raw_text.split():
		if lemmatizer(w) in Wish_61:
			WishTermNum += 1
	data.WishTerm = WishTermNum

@register
def ne2f(data, root):
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
def struc2f(self,data, root):
	struc = [0] * (len(self.feature_list) + 1)

	for _, f in enumerate(self.feature_list):
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
def labelNum2f(data, root):
	'''
	calculate the number of different label in first two level
	'''
	labelNum = [0]*len(self.labelList)
	lst1 = [e.edge_label for e in root.next_nodes]
	for i,id1 in enumerate(lst1):
		try:
			idx = self.labelList.index(id1)
		except ValueError:
			idx = 0
		labelNum[idx] += 1
		lst2 = [e.edge_label for e in root.next_nodes[i].next_nodes]
		for j in lst2:
			try:
				idx = self.labelList.index(j)
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


def find_feature(self, root):
	self.labelList.add(root.edge_label)

	if not root.next_nodes:
		r = treenode(root.ful_name)
		self.feature_list.append(r)
		return
	#Firstly generate feature with pred and two of its children

	# FirstLevelTuple = [(c1.edge_label, c2.edge_label) for c1 in root.next_nodes for c2 in root.next_nodes if c1 is not c2 and (c2.edge_label, c1.edge_label) not in FirstLevelTuple]
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
		if h  not in self.hashTable:
			self.hashTable.add(h)
			FirstLevelStruc.append(r)

	#Secondly, generate feature with path from pred to its leaf
	PathStruc = []
	for c1 in root.next_nodes:
		if not c1.next_nodes:
			self.labelList.add(c1.edge_label)
			r = treenode(root.ful_name)
			# n1 = treenode()
			# r.next_nodes.append(n1)
			r.next_edges.append(c1.edge_label)
			h = myHash(r)
			if h not in self.hashTable:
				self.hashTable.add(h)
				PathStruc.append(r)
			continue
		for c2 in c1.next_nodes:
			self.labelList.add(c2.edge_label)
			n1 = treenode()
			n1.next_edges.append(c2.edge_label)
			r = treenode(root.ful_name)
			r.next_nodes.append(n1)
			r.next_edges.append(c1.edge_label)
			h = myHash(r)
			if h not in self.hashTable:
				self.hashTable.add(h)
				PathStruc.append(r)
	
	self.feature_list = self.feature_list + FirstLevelStruc + PathStruc

def Process(subGraphs, TrainData):
	# subGraphs = None
	# with open('./tuple.pkl','rb') as p:					#！
	# 	TrainData = pickle.load(p)
	# if mode == '-train':
	# 	with open('../feature_extract/tuple.pkl','rb') as p:      #!
	# 		subGraphs = pickle.load(p)
	# elif mode == '-predict':
	# 	with open('./'+filename.split('/')[-1]+'.tp.pkl','rb') as p:    #！
	# 		subGraphs = pickle.load(p)
	# else:
	# 	raise Exception('Invalid Parameter')
	# root_list = []
	# # extract feature

	counter = 0
	for s in TrainData:
		amr = s.graph
		amr_nodes_acronym, root = amr_reader.amr_reader(amr)
		root_list.append(root)
		self.pred_list.add(root.ful_name)

		if s.annotation == 1.0:
			counter += 1
			find_feature(root)
	print(counter)

	self.labelList = ['MISC']+list(self.labelList)

	#compute feature
	for i,s in enumerate(subGraphs):
		root = root_list[i]
		for f in self.feature_funcs:
			f(s, root)
	
	attrs = [attr for attr in dir(subGraphs[0]) if attr not in dir(subGraph)]

	for a in ['graph','articleId','sentenceId','annotation','labelNum','raw_text','struc']:
		attrs.remove(a)
	print(attrs,'labelNum | struc')
	traindata = []
	for e in subGraphs:
		lst = [getattr(e,a) for a in attrs] + e.labelNum + e.struc + [e.annotation]
		traindata.append(lst)
	
	traindata = np.array(traindata,dtype = 'float64')
	return traindata
	
	# if mode == '-train':
	# 	with open('./traindata.pkl', 'wb') as p:
	# 		pickle.dump(traindata,p)
	# elif mode == '-predict':
	# 	with open('./'+filename.split('/')[-1]+'.ft.pkl','wb') as p:
	# 		pickle.dump(traindata,p)
	# else:
	# 	raise Exception('Invalid Parameter')

if __name__ == '__main__':
	mode = sys.argv[1]
	filename = sys.argv[2]
	main(mode, filename)