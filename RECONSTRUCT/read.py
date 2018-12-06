import xlrd
import re
from Record import record
import pickle


def read():
	with open('./text/subgraph.txt', 'r', encoding='utf8') as f:
		subgraph = f.read().split('-' * 50 + '\n')[2:]
	dic = {}
	for s in subgraph:
		lines = s.strip().split('\n')
		lineno = 0
		for line in lines:
			if line[0] == '#':
				if line[2:6] == "::id":
					SentenceId = line.split(' ')[2]
				if line[2:7] == "::snt":
					raw_text = line[8:]
				lineno += 1
			else:
				break
		graph = '\n'.join(lines[lineno:])
		if not graph:
			continue
		graphList = re.split("\n\n", graph)
		for g in graphList:
			graph_id = re.search("\((\d+)\)", g)

			if not graph_id:
				print("Error")
				exit(1)
			else:
				end = graph_id.end()
				graph_id = int(graph_id.group(1))
				graph = g[end:]
				r = record(raw_text, graph, SentenceId, graph_id, 0)
				key = str(SentenceId) + '-'+str(graph_id)
				dic[key] = r


	filename = "./text/annotation_naacl.xlsx"
	annotation = xlrd.open_workbook(filename).sheets()[3]
	maxrows = annotation.nrows
	lineno = 0
	trainset = []
	while lineno < maxrows:
		line = annotation.row(lineno)
		lineno+= 1
		if line[1].value:
			SentenceId = line[1].value
			continue
		if not line[2].value:
			continue
		graph_id = int(line[2].value)
		key = str(SentenceId) + '-' + str(graph_id)
		if key not in dic:
			print("Error: can not find the record")
			exit(1)
		if line[3].value == 'yes' or line[3].value == 'y':
			label = 1
			dic[key].annotation = label
			trainset.append(dic[key])
		elif line[3].value == 'no'or line[3].value == 'n':
			label = 0
			dic[key].annotation = label
			trainset.append(dic[key])
		elif not line[3].value:
			continue
	print(len(trainset))
	with open('./data/first_data.pkl', 'wb') as p:
		pickle.dump(trainset, p)

if __name__ == '__main__':
	read()



