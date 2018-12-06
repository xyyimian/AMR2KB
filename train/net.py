import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
	def __init__(self, in_d, hid_d, out_d):
		super(Net,self).__init__()
		self.layer1 = nn.Sequential(nn.Linear(in_d, hid_d), nn.ReLU(True))
		# self.layer2 = nn.Sequential(nn.Linear(hid1_d, hid2_d), nn.ReLU(True))
		self.layer2 = nn.Linear(hid_d, out_d)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		# x = self.layer3(x)
		return x