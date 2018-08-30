import sys
import math
import argparse
from random import randint

def getParser():
	parser = argparse.ArgumentParser(description='Input data points')
	parser.add_argument('--train', help='Training Data')
	parser.add_argument('--test', help='Testing Data')
	return parser

args = vars(getParser().parse_args(sys.argv[1:]))
trainfile = args['train']
testfile = args['test']

############ Training #####################################
text = {}
docs = {}
vocabulary = set()
numExamples = 0;
for line in open(trainfile):
	numExamples += 1
	doc = line.split()
	cat = doc[0]
	if (text.get(cat)) == None:
		text[cat] = {}
		docs[cat] = 0.0;
	docs[cat] += 1	
	for i in range(1,len(doc)):
		word = doc[i]
		if word not in vocabulary:
			vocabulary.add(word)
		if ((text[cat]).get(word)) == None:
			((text[cat])[word]) = 0.0
		(text[cat])[word] += 1.0


########### a. Naive Bayes Testing  #######################
docs1 = {};
for key in docs:
	docs1[key] = {}
	for key1 in docs:
		(docs1[key])[key1] = 0
totalcount = 0.0
correctcounta = 0.0
for line in open(testfile):
	v = -sys.maxsize
	list1 = line.split()
	for key in docs:
		p = math.log(docs[key]/numExamples)
		n = sum(text[key].values())
		for k in range(1,len(list1)):
			if ((text[key]).get(list1[k]) == None):
				nk = 0.0
			else:
				nk = (text[key])[list1[k]]
			p += math.log((nk+1)/(n+len(vocabulary)))
		if (p > v):
			v = p
			maxCat = key
	totalcount += 1
	(docs1[list1[0]])[maxCat] += 1	
	if (maxCat == list1[0]):
		correctcounta += 1

print ("")
print ("a) Naive Bayes Classifier : ")
print ("Correctly classified Groups : ",correctcounta)
print ("Total No. of Groups : ",totalcount)
print ("Accuracy : ",(correctcounta/totalcount)*100,"%\n")

############ b. Random Assignment ###########################
correctcountba = 0.0
for line in open(testfile):
	list1 = line.split()
	l = randint(0,len(docs)-1)
	cat = list(docs)[l]
	if (cat == list1[0]):
		correctcountba += 1

print ("b.1) Random Baseline")
print ("Correctly classified Groups : ",correctcountba)
print ("Total No. of Groups : ",totalcount)
print ("Accuracy : ",(correctcountba/totalcount)*100,"%\n")

maxD = 0
keyD = ""
for key in docs:
	if (docs[key] > maxD):
		maxD = docs[key]
		keyD = key
correctcountbb = 0
for line in open(testfile):
	list1 = line.split()
	cat = keyD
	if (cat == list1[0]):
		correctcountbb += 1

print ("b.2) Majority Baseline")
print ("Correctly classified Groups : ",correctcountbb)
print ("Total No. of Groups : ",totalcount)
print ("Accuracy : ",(correctcountbb/totalcount)*100,"%\n")

############ c. Confusion Matrix #############################
print ("c) Confusion Matrix")
print (docs1)