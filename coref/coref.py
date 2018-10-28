# -*- coding: utf-8 -*-
import re
import json
import sys
import os
def main(filename):

# firstly, read the raw text and write it into jsonl	
	with open(filename,'r') as f:
		text = f.read()	

	documents = text.split('#'*50)
	raw_text = open('raw_text.jsonl','w')
	for document in documents:
		document = ' '.join(document.strip().split('\n\n'))
		if not document:
			continue
		d = {}
		d['document'] = document
		jsondoc = json.dumps(d)
		raw_text.write(jsondoc)
		raw_text.write('\n')
	raw_text.close()




# secondly, use sys.cmd to call allennlp
	
	os.system("allennlp predict --silent ./coref-model-2018.02.05.tar.gz ./raw_text.jsonl --output-file output.jsonl")

#read the output file and use json.loads
	
	with open('output.jsonl','r') as f:
		text = f.read()
	f.close()
	resolutions = text.split('\n')
	for resolution in resolutions:
		if not resolution:
			continue
		data = json.loads(resolution)
		document = data['document']
		clusters = data['clusters']
		corefs = []
		for i in clusters:
			coref = []
			for j in i:
				s = document[slice(j[0],j[1]+1)]
				coref.append(" ".join(s))
			corefs.append(coref)
		for i in corefs:
			print(i)
			print('--------')
		print('#'*20)


if __name__ == '__main__':
	# filename = sys.argv[1]
	filename = "extracted_enwikinews_amr_sample_10.txt"
	main(filename)

