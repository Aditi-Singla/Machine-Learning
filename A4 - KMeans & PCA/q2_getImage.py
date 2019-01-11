import numpy as np
import os
import random
import time
import sys
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib.image as mpimg
from collections import Counter
from scipy import misc
# import scipy.misc


def getImage(filename):
	l1 = np.genfromtxt("allfiles_lfw.txt",dtype="str")
	if filename in l1:
		index = (l1.tolist()).index(filename)
		l11 = np.loadtxt("projections_lfw.txt")
		img = l11[index]
		eigen = np.loadtxt("eigen_lfw.txt")
		img = np.dot(eigen,img)
		misc.toimage(img.reshape(50,37)).save(filename.replace(".jpg","_l.jpg"))
		
	l2 = np.genfromtxt("allfiles_orl.txt",dtype="str")
	if filename in l2:
		index = (l2.tolist()).index(filename)
		l21 = np.loadtxt("projections_orl.txt")	
		img = l21[index]
		eigen = np.loadtxt("eigen_orl.txt")
		img = np.dot(eigen,img)
		misc.toimage(img.reshape(112,92)).save(filename.replace(".pgm","_l.pgm"))
		
getImage(sys.argv[1])