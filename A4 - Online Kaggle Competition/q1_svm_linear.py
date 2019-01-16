import time
import pickle
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
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


########### PCA Analysis ###################

print ("Training...")
pca = PCA(n_components=250)
pca.fit(Xtrain)
X = pca.transform(Xtrain)

############################################

print ("SVM training...")
mcsvm = svm.LinearSVC()
mcsvm = mcsvm.fit(X, Ytrain)

print ("Writing to file...")
pickle.dump(mcsvm,open("model_svm_linear","wb"))

print ("Testing....")
score = mcsvm.score(X,Ytrain)
print "Accuracy : ",(score*100),"%\n"

def crossValidate(data,labels,n):
	start = 	time.time()
	scores = cross_val_score(mcsvm,data,labels,cv=n)
	end = time.time()
	return (sum(scores)/n,end-start)

print "Cross Validation..."
(acc,timetaken) = crossValidate(X,Ytrain,5)
print "Accuracy for cross-validation : ",acc,"%"
print "Time taken : ",timetaken,"s"