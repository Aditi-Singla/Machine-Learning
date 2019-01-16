import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import tree
import pickle
import time

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

clf = tree.DecisionTreeClassifier(min_samples_split=10,min_samples_leaf=10,max_depth=50)

clf = clf.fit(Xtrain, Ytrain)

pickle.dump(clf,open("model_dtrees","wb"))
score = clf.score(Xtrain,Ytrain)
print "Accuracy : ",(score*100),"%\n"

def crossValidate(data,labels,n):
	start = time.time()
	scores = cross_val_score(clf,data,labels,cv=n)
	end = time.time()
	return (sum(scores)/n,end-start)

(acc,timetaken) = crossValidate(Xtrain,Ytrain,5)
print "Accuracy for cross-validation : ",acc,"%"
print "Time taken : ",timetaken,"s"