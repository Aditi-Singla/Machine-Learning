import sys
import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

Xtest = []

with open("test.csv") as fp:
	for line in fp:
		l = line.split(" ")
		arr = np.zeros(4125)
		indices = []
		nums = []
		for i in range(0,len(l)):
			l1 = l[i].split(":")
			indices.append(int(l1[0]))
			nums.append(float(l1[1]))
		np.put(arr,indices,nums)
		Xtest.append(arr)

clf = pickle.load(open(sys.argv[1]))
ans = clf.predict(Xtest)

outFile = open(sys.argv[2],'w')
outFile.write("ID,TARGET\n")
for i in range(len(ans)):
	outFile.write(str(i)+","+str(ans[i])+"\n")	
outFile.close()