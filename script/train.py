import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim     
from torch.autograd import Variable 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
sys.path.insert(0, '/home/bo/research/xprotocol')
import io

from model.CNN_based_Model import CNN_based_Model

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

data_train = np.load('../data/preprocessed/proposed/data_train.npy').item()
data_test = np.load('../data/preprocessed/proposed/data_test.npy').item()
label_train = np.load('../data/preprocessed/proposed/label_train.npy').item()
label_test = np.load('../data/preprocessed/proposed/label_test.npy').item()

np.load = np_load_old

def train_key(key, batch_size = 50, no_of_epochs = 20, no_of_hidden_units = 256, no_class = 2):

	x_train = np.array(data_train[key])
	y_train = np.array(label_train[key])
	x_test = np.array(data_test[key])
	y_test = np.array(label_test[key])
	print('Data loaded')

	# calculate number of each class
	no_of_class0, no_of_class1 = sum(y_train==0),sum(y_train==1)

	model = CNN_based_Model(no_of_hidden_units,no_class,[1.0/no_of_class0,1.0/no_of_class1])

	model.cuda()

	opt = 'adam'
	LR = 0.001
	if(opt=='adam'):
		optimizer = optim.Adam(model.parameters(), lr=LR)
	elif(opt=='sgd'):
		optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

	L_Y_train = len(y_train)
	L_Y_test = len(y_test)

	model.train()

	train_loss = []
	train_accu = []
	test_accu = []

	for epoch in range(no_of_epochs):

		# training
		model.train()

		epoch_loss = 0.0

		epoch_counter = 0

		epoch_acc_class = [0.0]*no_class
		epoch_counter_class = [0]*no_class

		time1 = time.time()

		I_permutation = np.random.permutation(L_Y_train)

		for i in range(0, L_Y_train, batch_size):
			
			x_input = x_train[I_permutation[i:i+batch_size]]
			y_input = y_train[I_permutation[i:i+batch_size]]

			data = Variable(torch.FloatTensor(x_input)).cuda()
			target = Variable(torch.LongTensor(y_input)).cuda()

			optimizer.zero_grad()
			loss, pred = model(data,target)
			loss.backward()

			optimizer.step()   # update weights

			acc = pred.eq(target).sum().cpu().data.numpy()

			epoch_loss += loss.data.item()
			epoch_counter += batch_size

			for c in range(no_class):
				mask = torch.zeros_like(pred) + c
				tmp = torch.eq(pred,mask) & torch.eq(pred,target)
				epoch_acc_class[c] += tmp.sum().cpu().data.numpy()
				epoch_counter_class[c] += torch.eq(target,mask).sum().cpu().data.numpy()

			assert(sum(epoch_counter_class)==epoch_counter)

		epoch_loss /= (epoch_counter/batch_size)

		for c in range(no_class):
			epoch_acc_class[c] /= epoch_counter_class[c]
		epoch_acc = np.mean(epoch_acc_class)

		train_loss.append(epoch_loss)
		train_accu.append(epoch_acc)

		if (epoch+1)%20==0:
			print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss,epoch_acc_class,epoch_counter_class)


		# ## test
		model.eval()

		epoch_loss = 0.0

		epoch_counter = 0

		epoch_acc_class = [0.0]*no_class
		epoch_counter_class = [0]*no_class

		time1 = time.time()

		I_permutation = np.random.permutation(L_Y_test)

		for i in range(0, L_Y_test, batch_size):

			x_input = x_test[I_permutation[i:i+batch_size]]
			y_input = y_test[I_permutation[i:i+batch_size]]

			data = Variable(torch.FloatTensor(x_input)).cuda()
			target = Variable(torch.LongTensor(y_input)).cuda()

			with torch.no_grad():
			    loss, pred = model(data,target)

			acc = pred.eq(target).sum().cpu().data.numpy()
			epoch_loss += loss.data.item()
			epoch_counter += batch_size

			for c in range(no_class):
				mask = torch.zeros_like(pred) + c
				tmp = torch.eq(pred,mask) & torch.eq(pred,target)
				epoch_acc_class[c] += tmp.sum().cpu().data.numpy()
				epoch_counter_class[c] += torch.eq(target,mask).sum().cpu().data.numpy()

			assert(sum(epoch_counter_class)==epoch_counter)

		epoch_loss /= (epoch_counter/batch_size)

		for c in range(no_class):
			epoch_acc_class[c] /= epoch_counter_class[c]
		epoch_acc = np.mean(epoch_acc_class)

		test_accu.append(epoch_acc)

		if (epoch+1)%20==0:
			print("  ", "%.2f" % (epoch_acc*100.0),epoch_acc_class,epoch_counter_class)

		# test baseline


if __name__ == "__main__":
	keys = list(data_train.keys())
	for key in keys:
		print(key)
		train_key(key,no_of_epochs=100)