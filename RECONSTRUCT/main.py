import pickle
import feature
from Record import record as subGraph

if __name__ == '__main__':
	with open('train_rec.pkl','rb') as p:
		train_rec = pickle.load(p)
		feature.m_train(train_rec)