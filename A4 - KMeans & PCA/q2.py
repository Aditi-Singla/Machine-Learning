import numpy as np
import os
import random
import time
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib.image as mpimg
from collections import Counter
from scipy import misc
# import scipy.misc


##########################################
############ Average Image ###############
##########################################

print "\n-------------------------"
print "Reading files...\n"

# DATASET 1
print "Folder 1 : Lfw_Easy"

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

X_lfw = []
Y_lfw = []
rowLfw = 0
colLfw = 0
counterlfw = 0
mapLfw = dict()
allFilesLfw = [] 
rootDir = './lfw_easy/'
for dirName, subdirList, fileList in os.walk(rootDir):
	for fname in fileList:
		ext = os.path.splitext(fname)[-1].lower()
		if (ext == ".pgm" or ext == ".jpg"):
			tmp =  dirName
			if tmp not in mapLfw.keys():
				counterlfw += 1
				mapLfw[tmp] = counterlfw
			Y_lfw.append(mapLfw[tmp])
			allFilesLfw.append(dirName+'/'+fname)
			img = misc.imread(dirName+'/'+fname, mode = 'F')
			rowLfw = len(img)
			colLfw = len(img[0])
			X_lfw.append(img.flatten())
Y_lfw = np.array(Y_lfw)
X_lfw = np.array(X_lfw)
mu_lfw = np.transpose(np.mean(X_lfw,axis=0))
misc.imshow(mu_lfw.reshape(rowLfw,colLfw))

# DATASET 2
print "Folder 2 : Orl_faces"

X_orl = []
Y_orl = []
rowOrl = 0
colOrl = 0
counterorl = 0
mapOrl = dict()
allFilesOrl = [] 
rootDir = './orl_faces/'
for dirName, subdirList, fileList in os.walk(rootDir):
	for fname in fileList:
		ext = os.path.splitext(fname)[-1].lower()
		if (ext == ".pgm" or ext == ".jpg"):
			tmp =  dirName
			if tmp not in mapOrl.keys():
				counterorl += 1
				mapOrl[tmp] = counterorl
			Y_orl.append(mapOrl[tmp])
			allFilesOrl.append(dirName+'/'+fname)
			img = mpimg.imread(dirName+'/'+fname)
			rowOrl = len(img)
			colOrl = len(img[0])
			X_orl.append(img.flatten())
Y_orl = np.array(Y_orl)
X_orl = np.array(X_orl)
mu_orl = np.transpose(np.mean(X_orl,axis=0))
misc.imshow(mu_orl.reshape(rowOrl,colOrl))


##########################################
############# PCA Analysis ###############
##########################################

print "\n-------------------------"
print "PCA Analysis..."

############## Main Function ###################

# Making the data Zero Mean
X_orl1 = X_orl - mu_orl
X_orl1 /= X_orl1.std(axis=0)
X_lfw1 = X_lfw - mu_lfw
X_lfw1 /= X_lfw1.std(axis=0)

# Principal Component Analysis

# DATASET 1
U1, S1, V1 = np.linalg.svd(X_lfw1, full_matrices=True)
eigenLfw = np.transpose(V1)[:,0:50] 
Z_lfw = np.dot(X_lfw1,eigenLfw)
Z_lfw /= Z_lfw.std(axis=0)

np.savetxt('allfiles_lfw.txt',allFilesLfw,fmt='%s')
np.savetxt('original_lfw.txt',X_lfw)
np.savetxt('originalNorm_lfw.txt',X_lfw1)
np.savetxt('eigen_lfw.txt',eigenLfw)
np.savetxt('labels_lfw.txt',Y_lfw)
np.savetxt('projections_lfw.txt',Z_lfw)

# DATASET 2
U, S, V = np.linalg.svd(X_orl1, full_matrices=True)
eigenOrl = np.transpose(V)[:,0:50] 
Z_orl = np.dot(X_orl1, eigenOrl)
Z_orl /= Z_orl.std(axis=0)

np.savetxt('allfiles_orl.txt',allFilesOrl,fmt='%s')
np.savetxt('original_orl.txt',X_orl)
np.savetxt('originalNorm_orl.txt',X_orl1)
np.savetxt('eigen_orl.txt',eigenOrl)
np.savetxt('labels_orl.txt',Y_orl)
np.savetxt('projections_orl.txt',Z_orl)



##########################################
############## Extra Tasks ###############
##########################################

print "\n------------------------------"
print "Eigen Faces..."

print "\nLFW Faces : Top 5 Principal Components"
for i in xrange(5):
	print "Face ",(i+1)
	maxNum = np.amax(eigenLfw[:,i])
	minNum = np.amin(eigenLfw[:,i])
	misc.toimage(((eigenLfw[:,i]-minNum).reshape(rowLfw,colLfw))*(200.0/maxNum)).save("lfw"+str(i+1)+".png")

print "\nPRL Faces : Top 5 Principal Components"
for i in xrange(5):
	print "Face ",(i+1)
	maxNum = np.amax(eigenOrl[:,i])
	minNum = np.amin(eigenOrl[:,i])
	misc.toimage(((eigenOrl[:,i]-minNum).reshape(rowOrl,colOrl))*(255.0/maxNum)).save("orl"+str(i+1)+".png")


##############################################
######## Finding the accuracy  ###############
##############################################

###########  Helper Functions ############

def crossValidate(data,labels,n):
	mcsvm = svm.SVC()
	start = time.time()
	scores = cross_val_score(mcsvm,data,labels,cv=n)
	end = time.time()
	return (sum(scores)/n,end-start)

##########################################

print "\n------------------------------"
print "\nCross Validation Accuracies using Scikit SVM...."

# DATASET 1

print "\nLFW Faces : "

print "Before...."
(acc,timetaken) = crossValidate(np.loadtxt("original_lfw.txt"),np.loadtxt("labels_lfw.txt"),10)
print "Accuracy : ",acc,"%"
print "Time taken : ",timetaken,"s"

print "\nAfter Normalisation...."
(acc,timetaken) = crossValidate(np.loadtxt("originalNorm_lfw.txt"),np.loadtxt("labels_lfw.txt"),10)
print "Accuracy : ",acc,"%"
print "Time taken : ",timetaken,"s"

print "\nAfter...."
(acc,timetaken) = crossValidate(np.loadtxt("projections_lfw.txt"),np.loadtxt("labels_lfw.txt"),10)
print "Accuracy : ",acc,"%"
print "Time taken : ",timetaken,"s"

# DATASET 2

print "\nORL Faces : "

print "Before...."
(acc,timetaken) = crossValidate(np.loadtxt("original_orl.txt"),np.loadtxt("labels_orl.txt"),10)
print "Accuracy : ",acc,"%"
print "Time taken : ",timetaken,"s"

print "\nAfter Normalisation...."
(acc,timetaken) = crossValidate(np.loadtxt("originalNorm_orl.txt"),np.loadtxt("labels_orl.txt"),10)
print "Accuracy : ",acc,"%"
print "Time taken : ",timetaken,"s"

print "\nAfter...."
(acc,timetaken) = crossValidate(np.loadtxt("projections_orl.txt"),np.loadtxt("labels_orl.txt"),10)
print "Accuracy : ",acc,"%"
print "Time taken : ",timetaken,"s"