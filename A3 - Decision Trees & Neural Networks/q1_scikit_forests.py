import sys
from sklearn.ensemble import RandomForestClassifier

with open(sys.argv[1]) as f:
	file = f.readlines()

Xtrain = []
Ytrain = []

for i in range(1,len(file)):
	Xtrain.append(file[i].split(",")[:-1])
	Ytrain.append(file[i].split(",")[-1])

clf = RandomForestClassifier(n_estimators=30,max_features=30,bootstrap=True)
clf = clf.fit(Xtrain, Ytrain)

with open(sys.argv[2]) as f:
	file = f.readlines()

Xtest = []
Ytest = []

for i in range(1,len(file)):
	Xtest.append(file[i].split(",")[:-1])
	Ytest.append(file[i].split(",")[-1])

ans = clf.predict(Xtest)

correctCount = 0
for i in range(len(Ytest)):
	if (ans[i]==Ytest[i]):
		correctCount += 1

print float(correctCount)/len(Ytest)