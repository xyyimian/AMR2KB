# -*- coding:utf-8 -*-
import pickle
import sys
sys.path.append('.')
import feature
from Record import record
import Record
import numpy as np
from net import Net
import torch
import torch.nn.functional as F
import re
import numpy



def predict(records):
	np.random.shuffle(records)
	data = torch.from_numpy(records)

	device = torch.device("cpu")

	X = data[:,:-1]
	l = len(X)

	net = Net(X.size(1), 128, 2)
	net = torch.load('./models/classifier.pth')
	net.double()
	net.to(device)

	y_pred = net(X)
	y_pred = torch.max(F.softmax(y_pred),1)[1]
	binder = zip(records, y_pred)
	return binder

def main(filename):


	g = open('./text/'+filename+'.sbg','r', encoding = 'utf8')
	sbg_list = re.split('-'*50+'\n', g.read().strip())[1:]
	records = Record.ProduceRecords(sbg_list)
	print("Processing %d records in total" % len(records))

	with open('./intermediate/feature_template.pkl','rb') as p:
		featCombination = pickle.load(p)

	data, idmap = feature.process(records, featCombination)
	binder = predict(data)
	with open('./text/' + filename + '.kng','w', encoding = 'utf8') as f:
		knowledge_list = []
		cn = 0
		for e in binder:
			cn += 1
			r, y_pred = e
			if y_pred == 1:
				uniid = idmap[str(int(r[0]))]
				articleId, sentenceId = uniid.split('-')
				graph  = [record.graph for record in records if record.articleId == articleId and record.sentenceId == sentenceId]
				assert len(graph) == 1
				knowledge_list.append((articleId, sentenceId, graph[0]))
		knowledge_list = sorted(knowledge_list, key = lambda x:(int(x[0]),int(x[1])))
		for knowledge in knowledge_list:
			f.write(knowledge[0]+'-'+knowledge[1]+'\n'+knowledge[2]+'\n')

	g.close()

if __name__ == '__main__':
	filename = sys.argv[1]
	filename = filename.split('/')[-1]
	print(filename)
	main(filename)


