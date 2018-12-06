#	-*- coding:utf-8 -*-
import xlrd
import csv
import re
import pickle
import sys
import random
TEST_DATASET_SIZE = 500


class subGraph():
	def __init__(self, raw_text, graph, articleId, sentenceId, annotation):
		self.raw_text = raw_text
		self.graph = graph
		self.articleId = articleId
		self.sentenceId = sentenceId
		self.annotation = annotation

	def __repr__(self):
		return "{}\t{}\t{}\t{}\t{}".format(self.raw_text, self.articleId, self.sentenceId, self.graph, self.annotation)

class data2tuple:
	def combine(subgraph, annotationPath):
		'''
		This function is used for training data specifically
		'''
		annotation = xlrd.open_workbook(annotationFilePath).sheets()[0]
		maxrows = annotation.nrows

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
				subGraphs.append(subGraph(raw_text,graph,aid,sid,y))
		return subGraphs

	def ProduceTuple(subgraph):
		for s in subgraph:
			line = s.split('\n')
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
				r = re.search("\((\d+)\)", e)
				if not r:
					print('sb')
				ind = int(r.group(1))  # a integer
				graph = e[r.end(1) + 1:]
				subGraphs.append(subGraph(raw_text,graph,None,None,None))
		return subGraphs


def main(mode, filename):
	lino = 0
	subGraphs = []
	converter = data2tuple()
	if mode == '-train':
		annotationFilePath = "../feature_extract/annotation.xlsx"
		subGraphPath = '../subgraph/PART-amr-release-1.0-training-proxy-subgraph.txt'
		converter.combine(subgraph, annotationFilePath)

		print("Processing %d tuples in total" % len(subGraphs))
		with open('./tuple.pkl','wb') as p:		#the output file need to be open as wb if list
			pickle.dump(subGraphs,p)

	elif mode == '-predict':
		subgraphFilePath = './subgraph/'+filename.split('/')[-1]+'.sbg'
		subgraph = open(subgraphFilePath, 'r').read().split('-' * 50 + '\n')[1:]
		subGraphs = converter.ProduceTuple(subgraph)

		print("Processing %d tuples in total" % len(subGraphs))
		with open('./feature_extract/'+filename.split('/')[-1]+'.tp.pkl','wb') as p:
			pickle.dump(subGraphs, p)
	else:
		raise Exception('Invalid Parameter')

if __name__ == '__main__':
	mode = sys.argv[1]
	filename = sys.argv[2]
	main(mode, filename)


