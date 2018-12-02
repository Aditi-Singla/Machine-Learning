import random
import numpy as np
import time
import math

###################################################
##############  Helper functions  #################
###################################################

def sigmoid(m):
	g = m.shape
	ans = np.matrix(np.zeros(g))
	ans = 1/(1 + np.exp(np.multiply(-1, m)))
	return ans

def relu(m):
	g = m.shape
	ans = np.matrix(np.zeros(g))
	ans = np.log(1 + np.exp(m))
	return ans		

######## Compute the label given an input and the learnt model  ########

def compute_label(x, w1, w2, b1, b2):
	net1 = w1*x + b1
	v1 = sigmoid(net1)
	net2 = w2*v1 + b2
	v2 = sigmoid(net2)
	ans = np.argmax(v2, axis=0)
	return (ans[0,0] + 1)

def compute_label_relu(x, w1, w2, b1, b2):
	net1 = w1*x + b1
	v1 = relu(net1)
	net2 = w2*v1 + b2
	v2 = relu(net2)
	ans = np.argmax(v2, axis=0)
	return (ans[0,0] + 1)

###### Compute dW1, dW2, dB1, dB2, given an input, its label and a model ############

def get_grads(x, y, w1, w2, b1, b2):
	num_labels, hidden_layer_size = w2.shape
	net1 = w1*x + b1
	v1 = sigmoid(net1)
	net2 = w2*v1 + b2
	v2 = sigmoid(net2)
	
	del2 = np.matrix(np.zeros((num_labels,1)))
	del2[int(y)-1, 0] = 1
	del2 = v2 - del2
	dw21 = np.multiply(v2,(1-v2))
	dw21 = np.multiply(dw21,del2)

	dW2 = dw21*(np.transpose(v1))
	dB2 = del2	
	
	del1 = np.transpose(w2)*dw21 
	dw11 = np.multiply(v1,(1-v1))
	dw11 = np.multiply(dw11,del1)
	
	dW1 = dw11*(x.T)
	dB1 = del1
	
	return (dW1, dB1, dW2, dB2)

def get_grads_relu(x, y, w1, w2, b1, b2):
	num_labels, hidden_layer_size = w2.shape
	net1 = w1*x + b1
	v1 = relu(net1)
	net2 = w2*v1 + b2
	v2 = relu(net2)
	
	del2 = np.matrix(np.zeros((num_labels,1)))
	del2[int(y)-1, 0] = 1
	del2 = v2 - del2
	dw21 = sigmoid(net2)
	dw21 = np.multiply(dw21,del2)

	dW2 = dw21*(np.transpose(v1))
	dB2 = del2
	
	del1 = np.transpose(w2)*dw21 
	dw11 = sigmoid(net1)
	dw11 = np.multiply(dw11,del1)
	
	dW1 = dw11*(x.T)
	dB1 = del1
	
	return (dW1, dB1, dW2, dB2)	

###### Initialise the weight matrices randomly ###############

def rand_initialise_weight(lout, lin):
	eps_init = (np.sqrt(6))/(np.sqrt(lout) + np.sqrt(lin))
	ans = np.matrix(np.zeros((lout, lin)))
	np.random.seed(1)
	ans = (np.random.randn(lout, lin) * 2 * eps_init) - eps_init
	return ans

####### Train a model using Stochastic Gradient Descent #######

def train_model_sgd(data, w1, w2, b1, b2, alpha, num_iter):
	m, n = data.shape
	np.random.shuffle(data)
	X_train = data[:,0:(n-1)]
	y_train = data[:,n-1]
	
	print "Starting to train..."
	for j in range(num_iter):
		start = time.time()
		for i in range(m):
			x_temp = np.matrix(np.transpose(X_train[i,:]))
			y_temp = y_train[i,0]
			(w1_grad, b1_grad, w2_grad, b2_grad) = get_grads(x_temp, y_temp, w1, w2, b1, b2)
			w2 = w2 - (alpha*w2_grad)
			w1 = w1 - (alpha*w1_grad)
			b2 = b2 - (alpha*b2_grad)
			b1 = b1 - (alpha*b1_grad)

		print('Iteration: {}'.format(j+1))
		print "Time taken : ",(time.time()-start)
	return (w1, w2, b1, b2)


###################################################
#########  Read input file 'train.data'  ##########
###################################################

m_train = 0
n_train = 0
with open('connect_4/train.data') as fp:
	for line in fp:
		m_train += 1
		n_train = len(line.split(","))

train_data = np.matrix(np.zeros((m_train,n_train)))

with open('connect_4/train.data') as fp:
	row = 0
	for line in fp:
		a = line.split(",")
		for col in range(len(a)):
			c = a[col][0]
			train_data[row,col] = 0 if c=='0' else (1 if c=='1' or c == 'w' else (2 if c=='l' else 3))
		row += 1


###################################################
###############  Main Function  ###################
###################################################

input_layer_size = n_train - 1
num_labels = 3
hidden_layer_size = 100
alpha = .003
num_iter = 40

########## Initialise weights randomly ############
W1 = rand_initialise_weight(hidden_layer_size, input_layer_size)
W2 = rand_initialise_weight(num_labels, hidden_layer_size)

##############  Initialise biases  ################
B1 = rand_initialise_weight(hidden_layer_size, 1) #np.matrix(np.zeros((hidden_layer_size, 1)))
B2 = rand_initialise_weight(num_labels, 1) #np.matrix(np.zeros((num_labels, 1)))

############ Learning the model ##################

start_train = time.time()
(w1_out, w2_out, b1_out, b2_out) = train_model_sgd(train_data, W1, W2, B1, B2, alpha, num_iter)
print "Total Time taken : ",(time.time()-start_train)

raw_input("Press Enter to continue...")


##################################################
##########  Prediction on training set  ##########
##################################################

acc = 0
m, n = train_data.shape
X = train_data[:,0:n-1]
y = train_data[:,n-1]
for i in range(m):
	pred = compute_label(np.matrix(np.transpose(X[i,:])), w1_out, w2_out, b1_out, b2_out)	
	if(y[i,0] == pred):
		acc = acc + 1
print(acc)
print('Training set {} % ||'.format(float(acc)/m * 100))


##################################################
############  Prediction on test set  ############
##################################################

m_test = 0
n_test = 0
with open('connect_4/test.data') as fp:
	for line in fp:
		m_test += 1
		n_test = len(line.split(","))

test_data = np.matrix(np.zeros((m_test,n_test)))

with open('connect_4/test.data') as fp:
	row = 0
	for line in fp:
		a = line.split(",")
		for col in range(len(a)):
			c = a[col][0]
			test_data[row,col] = 0 if c=='0' else (1 if c=='1' or c == 'w' else (2 if c=='l' else 3))
		row += 1

acc = 0
m, n = test_data.shape
X = test_data[:,0:n-1]
y = test_data[:,n-1]
for i in range(m):
	pred = compute_label(np.matrix(np.transpose(X[i,:])), w1_out, w2_out, b1_out, b2_out)	
	if(y[i,0] == pred):
		acc = acc + 1
print(acc)
print('Testing set {} % ||'.format(float(acc)/m * 100))