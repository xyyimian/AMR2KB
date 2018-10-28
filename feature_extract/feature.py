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

feature_funcs = []
lemmatizer = partial(WordNetLemmatizer().lemmatize,pos = 'v')

class treenode:
	def __init__(self, value = ""):
		self.next_edges = [] 		#label
		self.next_nodes = []		#reference
		self.value = value



def register(feature_func):
	print('extracting feature: ',feature_func.__name__)
	feature_funcs.append(feature_func)
	return feature_func

@register
def pred2f(data, root):
	'''
	predicate class as feature
	'''
	global pred_list
	try:
		idx = list(pred_list).index(root.ful_name)
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
def struc2f(data, root):
	global feature_list

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







# @register
# def label2f(data, root):
# 	'''
# 	compute the simularities between sample and traindata
# 	'''

# 	global feature_list
# 	max = 0
# 	lst1 = [e.edge_label for e in root.next_nodes]
# 	if len(lst1) == 0:
# 		data.labelMark = max
# 		return
# 	for f in feature_list:
# 		mark = 0
# 		mark += sum([1 if e in f.next_edges else 0 for e in lst1]) / len(lst1)
# 		if mark == 0:
# 			continue
# 		numerator = 0
# 		denominator = 0
# 		for i in range(len(lst1)):
# 			try:
# 				id = f.next_edges.index(lst1[i])
# 			except ValueError:
# 				continue
# 			lst2 = [e.edge_label for e in root.next_nodes[i].next_nodes]
# 			denominator += len(lst2)
# 			numerator += sum([1 if e in f.next_nodes[id].next_edges else 0 for e in lst2])
# 		if denominator == 0:
# 			continue
# 		mark += 2 * numerator / denominator
# 		max = mark if mark > max else max
# 	data.labelMark = max

@register
def labelNum2f(data, root):
	'''
	calculate the number of different label in first two level
	'''
	global labelList
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




feature_list = []
pred_list = set()
labelList = set()
hashTable = set()


def myHash(treeroot):
	s = ""
	for i, e1 in enumerate(sorted(treeroot.next_edges)):
		s = s + str(i) + e1
		if not treeroot.next_nodes:
			continue
		for j, e2 in enumerate(sorted(treeroot.next_nodes[i].next_edges)):
			s = s + str(j) + e2
	return s


def find_feature(root):
	global feature_list
	global labelList
	global hashTable

	labelList.add(root.edge_label)

	if not root.next_nodes:
		r = treenode(root.ful_name)
		feature_list.append(r)
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


# def find_feature(root):
# 	global feature_list
# 	global labelList
# 	#find structure with one node

# 	r = treenode()
# 	labelList.add(root.edge_label)
# 	for c1 in root.next_nodes:
# 		labelList.add(c1.edge_label)
# 		n1 = treenode()
# 		r.next_edges.append(c1.edge_label)
# 		for c2 in c1.next_nodes:
# 			labelList.add(c2.edge_label)
# 			n2 = treenode()
# 			n1.next_edges.append(c2.edge_label)
# 			n1.next_nodes.append(n2)
# 		r.next_nodes.append(n1)
# 	feature_list.append(r)




def main(mode, filename):
	global feature_funcs
	global pred_list
	global labelList

	subGraphs = None
	with open('./tuple.pkl','rb') as p:					#！
		TrainData = pickle.load(p)
	if mode == '-train':
		with open('../feature_extract/tuple.pkl','rb') as p:      #!
			subGraphs = pickle.load(p)
	elif mode == '-predict':
		with open('./'+filename.split('/')[-1]+'.tp.pkl','rb') as p:    #！
			subGraphs = pickle.load(p)
	else:
		raise Exception('Invalid Parameter')
	root_list = []
	# extract feature

	counter = 0
	for s in TrainData:
		amr = s.graph
		amr_nodes_acronym, root = amr_reader.amr_reader(amr)
		root_list.append(root)
		pred_list.add(root.ful_name)

		if s.annotation == 1.0:
			counter += 1
			find_feature(root)
	print(counter)

	labelList = ['MISC']+list(labelList)

	#compute feature
	for i,s in enumerate(subGraphs):
		root = root_list[i]
		for f in feature_funcs:
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
	print(traindata.dtype)
	if mode == '-train':
		with open('./traindata.pkl', 'wb') as p:
			pickle.dump(traindata,p)
	elif mode == '-predict':
		with open('./'+filename.split('/')[-1]+'.ft.pkl','wb') as p:
			pickle.dump(traindata,p)
	else:
		raise Exception('Invalid Parameter')

if __name__ == '__main__':
	mode = sys.argv[1]
	filename = sys.argv[2]
	main(mode, filename)