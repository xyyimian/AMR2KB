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
transform = transforms.Compose(transforms.ToTensor())

learning_rate = 0.001

def train():
	with open('./feature_extract/traindata.pkl','rb') as p:
		data = pickle.load(p)
	random.shuffle(data)
	data = torch.from_numpy(data)
	# data = data.long()
	device = torch.device("cpu")

	X = data[:,:-1]
	y = data[:,-1:].type(torch.LongTensor).squeeze(1)  #CrossEntropy just receive 1-D tensor, even [32,1] need to be squeenze

	l = len(X)
	s = floor(0.8 * l)

	X_train = X[:s]
	X_test = X[s:]
	y_train = y[:s]
	y_test = y[s:]


	net = Net(X.size(1), 128, 128, 2)
	net.double()
	net.to(device)	

	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr = learning_rate)

	for batch in range(500):
		running_loss = 0.0
		optimizer.zero_grad()

		y_pred = net(X_train)
		# print(y_pred)
		loss = loss_fn(y_pred, y_train)
		loss.backward()

		optimizer.step()
		running_loss = loss.item()
		print(running_loss)

	torch.save(net, './train/classifier.pth')

	correct = 0
	with torch.no_grad():
		outputs = net(X_test)
		# _, predicted = torch.max(outputs, 1)
		prediction = torch.max(F.softmax(outputs),1)[1]
		print(prediction)
		print('#' * 40)
		print(y_test)
		total = y_test.size(0)
		correct += (prediction == y_test).sum().item()
	print(correct)

	print('Accuracy of the network on the test dataset:%d %%' % (100*correct / total))

if __name__ == '__main__':
	train()