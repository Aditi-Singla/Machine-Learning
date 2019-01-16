import time
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

Xtrain = []
Ytrain = []

with open("train.csv") as fp:
	for line in fp:
		l = line.split(" ")
		Ytrain.append(int(l[0]))
		arr = np.zeros(4125)
		indices = []
		nums = []
		for i in range(1,len(l)):
			l1 = l[i].split(":")
			indices.append(int(l1[0]))
			nums.append(float(l1[1]))
		np.put(arr,indices,nums)
		Xtrain.append(arr)

gnb = GaussianNB()
gnb = gnb.fit(np.array(Xtrain), Ytrain)

pickle.dump(gnb,open("model_nb_gaussian","wb"))
score = gnb.score(np.array(Xtrain),Ytrain)
print "Accuracy : ",(score*100),"%\n"

def crossValidate(data,labels,n):
	start = time.time()
	scores = cross_val_score(gnb,data,labels,cv=n)
	end = time.time()
	return (sum(scores)/n,end-start)

(acc,timetaken) = crossValidate(np.array(Xtrain),Ytrain,5)
print "Accuracy for cross-validation : ",(acc*100),"%"
print "Time taken : ",timetaken,"s"