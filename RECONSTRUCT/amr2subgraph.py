#-*- coding:utf-8 -*-

from nominalization import *
import re
from models.Node import Node
import urllib
import copy
import sys
from constants import conjuctionList
from models.Sentence import Sentence
sys.setrecursionlimit(10000)


def generate_node_single(content, amr_nodes_content, amr_nodes_acronym):
	'''
	Generate Node object for single '()'

	:param str context:
	:param dict amr_nodes_content: content as key
	:param dict amr_nodes_acronym: acronym as key
	'''
	is_named_entity = False
	try:
		assert content.count('(') == 1 and content.count(')') == 1
	except AssertionError:
		raise Exception('Unmatched parenthesis')

	predict_event = re.search('(\w+)\s/\s(\S+)', content)
	if predict_event:
		acr = predict_event.group(1) # Acronym
		ful = predict_event.group(2).strip(')') # Full name
	else:
		acr, ful = '-', '-'

	# In case of :polarity -
	is_polarity = True if re.search(":polarity\s-", content) else False

	# :ARG ndoes
	arg_nodes = []
	nodes = re.findall(':\S+\s\S+', content)
	for i in nodes:
		i = re.search('(:\S+)\s(\S+)', i)
		role = i.group(1)
		concept = i.group(2).strip(')')
		if role == ':wiki' and is_named_entity:
			continue
		if role == ':polarity':
			continue
		if concept in amr_nodes_acronym:
			node = copy.copy(amr_nodes_acronym[concept])
			node.next_nodes = []
		# In case of (d / date-entity :year 2012)
		else:
			node = Node(name=concept, original_content= concept)
			amr_nodes_acronym[concept] = node
		node.edge_label = role
		arg_nodes.append(node)

	# Node is a named entity
	names = re.findall(':op\d\s\"\S+\"', content)
	if len(names) > 0:
		entity_name = ''
		for i in names:
			entity_name += re.match(':op\d\s\"(\S+)\"', i).group(1) + ' '
		entity_name = urllib.parse.unquote_plus(entity_name.strip())
		new_node = Node(name=acr, ful_name=ful, next_nodes=arg_nodes,parents = set(),
						entity_name=entity_name,
						polarity=is_polarity, content=content,original_content = content)
		amr_nodes_content[content] = new_node
		amr_nodes_acronym[acr] = new_node
	else:
		new_node = Node(name=acr, ful_name=ful, next_nodes=arg_nodes,parents = set(),
						polarity=is_polarity, content=content,original_content=content)
		amr_nodes_content[content] = new_node
		amr_nodes_acronym[acr] = new_node

def generate_nodes_multiple(content, amr_nodes_content, amr_nodes_acronym):
	'''
	Generate Node object for nested '()'

	:param str context:
	:param dict amr_nodes_content: content as key
	:param dict amr_nodes_acronym: acronym as key
	'''
	try:
		assert content.count('(') > 1 and content.count(')') > 1
		assert content.count('(') == content.count(')')
	except AssertionError:
		raise Exception('Unmatched parenthesis')

	#note that if we want to get complete content, we need to complete subgraphs in amr_nodes_content, however if we really complete it, the following can't be indexed in amr_nodes_content
	_content = content  #because content will gradually move its components, so use _content to back up for amr_nodes_content
	org = content  #original_content   difference between above is It don't remove :name
	arg_nodes = []
	is_named_entity = False

	# Remove existing nodes from the content, and link these nodes to the root
	# of the subtree
	for i in sorted(amr_nodes_content, key=len, reverse=True):
		if i in content:
			e = content.find(i)
			s = content[:e].rfind(':')
			role = re.search(':\S+\s', content[s:e]).group() # Edge label

			amr_nodes_content[i].edge_label = role.strip()
			if ':name' in role:
				is_named_entity = True
				ne = amr_nodes_content[i]
			else:
				 arg_nodes.append(amr_nodes_content[i])
			if ':name' not in role:
				org = org.replace(role + i,'',1)
			content = content.replace(role + i, '', 1)

	predict_event = re.search('\w+\s/\s\S+', content).group().split(' / ')
	if predict_event:
		acr = predict_event[0] # Acronym
		ful = predict_event[1] # Full name
	else:
		acr, ful = '-', '-'

	# In case of :polarity -
	is_polarity = True if re.search(":polarity\s-", content) else False

	nodes = re.findall(':\S+\s\S+', content)
	for i in nodes:
		i = re.search('(:\S+)\s(\S+)', i)
		role = i.group(1)
		concept = i.group(2).strip("()")
		if role == ':wiki' and is_named_entity:
			continue
		if role in [':polarity',':quant',':age',':value']:
			continue
		if concept in amr_nodes_acronym:
			node = copy.copy(amr_nodes_acronym[concept])
			content = content.replace(i.group(0),"")
		# In case of (d / date-entity :year 2012)
		else:
			node = Node(name=concept)
			amr_nodes_acronym[concept] = node
			# if re.search("\s+"+concept+"[^\d]", content, ):
			# 	content.replace(concept, )
		node.edge_label = role
		arg_nodes.append(node)

		# Named entity is a special node, so the subtree of a
		# named entity will be merged. For example,
		#     (p / person :wiki -
		#        :name (n / name
		#                 :op1 "Pascale"))
		# will be merged as one node.
		# According to AMR Specification, "we fill the :instance
		# slot from a special list of standard AMR named entity types".
		# Thus, for named entity node, we will use entity type
		# (p / person in the example above) instead of :instance

	if is_named_entity:
		# Get Wikipedia title:
		if re.match('.+:wiki\s-.*', content):
			wikititle = '-' # Entity is NIL, Wiki title does not exist
		else:
			m = re.search(':wiki\s\"(.+?)\"', content)
			if m:
				wikititle = urllib.parse.unquote_plus(m.group(1)) # Wiki title
			else:
				wikititle = '' # There is no Wiki title information

		new_node = Node(name=acr, ful_name=ful, next_nodes=arg_nodes, parents = set(),
						edge_label=ne.ful_name, is_entity=True,
						entity_type=ful, entity_name=ne.entity_name,
						wiki=wikititle, polarity=is_polarity, content=content, original_content = org)
		amr_nodes_content[_content] = new_node
		amr_nodes_acronym[acr] = new_node

	elif len(arg_nodes) > 0:
		new_node = Node(name=acr, ful_name=ful, next_nodes=arg_nodes, parents = set(),
						polarity=is_polarity, content=content, original_content=_content)
		amr_nodes_content[_content] = new_node
		amr_nodes_acronym[acr] = new_node
	for child in new_node.next_nodes:
		child.parents.add(new_node)

def split_amr(raw_amr, contents, amr_contents):
	'''
	Split raw AMR based on '()'

	:param str raw_amr:
	:param list contentss:
	'''
	if not raw_amr:
		return
	else:
		if raw_amr[0] == '(':
			contents.append([])
			for i in contents:
				i.append(raw_amr[0])
		elif raw_amr[0] == ')':
			for i in contents:
				i.append(raw_amr[0])
			amr_contents.append(''.join(contents[-1]))
			contents.pop(-1)
		else:
			for i in contents:
				i.append(raw_amr[0])
		raw_amr = raw_amr[1:]
		split_amr(raw_amr, contents, amr_contents)


def amr_reader(raw_amr):
	'''
	:param str raw_amr: input raw amr
	:return dict amr_nodes_acronym:
	:return list path:
	'''

	amr_contents = []
	amr_nodes_content = {} # Content as key
	amr_nodes_acronym = {} # Acronym as key
	path = [] # Nodes path
	dic = {}

	split_amr(raw_amr, [], amr_contents)
	#construct a dictionary to index subtree for diffrent root

	for i in amr_contents:
		if i.count('(') == 1 and i.count(')') == 1:
			generate_node_single(i, amr_nodes_content, amr_nodes_acronym)
	for i in amr_contents:
		if i.count('(') > 1 and i.count(')') > 1:
			generate_nodes_multiple(i, amr_nodes_content, amr_nodes_acronym)
	for i in amr_contents:
		if i.count('(') == 1 and i.count(')') == 1:
			revise_node(i, amr_nodes_content, amr_nodes_acronym)

	for subtree in amr_nodes_content.keys():
		pattern = re.compile("\:\S+\s(\S+?)[\n\)]+")
		sub = []
		for m in pattern.finditer(subtree):
			if m.group(1)[0] == '"':
				continue
			start, end = m.start(1), m.end(1)
			acronym = subtree[start:end]
			if acronym in amr_nodes_acronym:
				org = amr_nodes_acronym[acronym].original_content
				sub.append((start, end, org))
		sub = sorted(sub, key = lambda x : x[0])
		lastend = 0
		s = ""
		for start, end, org in sub:
			s += subtree[lastend:start] + org
			lastend = end
		s += subtree[lastend:]

		key = re.search('\((\S+)', s).group(1)
		dic[key] = s

	# The longest node (entire AMR) should be the root
	if not amr_nodes_content:
		print('sb')
	root = amr_nodes_content[sorted(amr_nodes_content, key=len, reverse=True)[0]]
	root.parents.add('@')
	return amr_nodes_acronym, root, dic

	#for all named entity, record their trace to the root
def getTrace(amr_nodes_acronym, dic):
	trace = set()
	for e in amr_nodes_acronym.values():
		if e.is_entity:
			temp = set()
			if '@' in e.parents:
				continue
			queue = list(e.parents)
			while queue:    #queue: node list; current: current node
				current = queue[-1]
				queue.pop()
				if '@' not in current.parents:
					queue += list(current.parents)

				if not current.is_entity and len(current.next_nodes) < 2:
					continue
				t = re.search('(\w+)-\d+',current.ful_name)
				if t:
					t = t.group(1)
				else:
					t = current.ful_name
				if t in conjuctionList:
					continue
				temp.add(current.name)

			if len(e.next_nodes) > 1:
				for a in e.next_nodes:
					try:
						dic[a.name] = e.original_content[:-1].strip()+'\n\t' + a.edge_label+' '+dic[a.name]+')'
						temp.add(a.name)
					except KeyError:
						print(a.name)
						print(e.name)
						continue
			else:

				#if e is not a bare entity
				attr = re.findall(':(\S+)',e.original_content)
				for a in attr:
					if a != 'name' and 'op' not in a:
						temp.add(e.name)
						break

			trace = trace.union(temp)
	return trace
	# ind = 0
	# for t in trace:
	#     ind += 1
	#     writer.write('('+str(ind)+')'+self.dic[t])
	#     writer.write('\n\n')
	# return amr_nodes_acronym, path



def retrieve_path(node, parent, path):
	'''
	Retrieve AMR nodes path

	:param Node_object node:
	:param str parent:
	:param list path:
	'''
	path.append((parent, node.name, node.edge_label))
	for i in node.next_nodes:
		retrieve_path(i, node.name, path)


def revise_node(content, amr_nodes_content, amr_nodes_acronym):
	'''
	In case of single '()' contains multiple nodes
	e.x. (m / moment :poss p5)

	:param str context:
	:param dict amr_nodes_content: content as key
	:param dict amr_nodes_acronym: acronym as key
	'''
	m = re.search('\w+\s/\s\S+\s+(.+)', content.replace('\n', ''))
	if m and ' / name' not in content and ':polarity -' not in content:
		arg_nodes = []
		acr = re.search('\w+\s/\s\S+', content).group().split(' / ')[0]
		nodes = re.findall('\S+\s\".+\"|\S+\s\S+', m.group(1))
		for i in nodes:
			i = re.search('(:\S+)\s(.+)', i)
			role = i.group(1)
			concept = i.group(2).strip(')')
			if concept in amr_nodes_acronym:
				node = copy.copy(amr_nodes_acronym[concept])
				node.next_nodes = []
			else: # in case of (d / date-entity :year 2012)
				node = Node(name=concept)
				amr_nodes_acronym[concept] = node
			node.edge_label = role
			arg_nodes.append(node)
		amr_nodes_acronym[acr].next_nodes = arg_nodes
		amr_nodes_content[content].next_nodes = arg_nodes


def process(raw_amrs):
	subgraph = ""
	for raw_amr in raw_amrs:
		subgraph += '-'*50 + '\n'
		raw_amr_lst = raw_amr.split('\n')
		text = []
		for e in raw_amr_lst:
			if e.startswith('#'):
				subgraph += e+'\n'
				continue
			else:
				text.append(e)
		raw_amr = '\n'.join(text)
		if not raw_amr:
			continue
		amr_nodes_acronym, path, dic= amr_reader(raw_amr)
		trace = getTrace(amr_nodes_acronym, dic)
		ind = 0
		for t in trace:
			ind += 1
			try:
				subgraph += '('+str(ind)+')'+dic[t]+'\n\n'
			except KeyError:
				print('sb')
	return subgraph

if __name__ == '__main__':
	filename = sys.argv[1]  #path to original text
	process(filename)