import numpy as np
import sys

outFile = open(sys.argv[2],'w')
with open(sys.argv[1]) as fp:
	b = True
	for line in fp:
		if (b):
			b = False
			l = line.split(",")
			n = len(l)
			outFile.write(str(n)+"\n")
			for i in range(n):
				l1 = l[i].split(":")
				if (l1[1] == "Continuous"):
					outFile.write("0 " + l1[0] + "\n")
				else:
					outFile.write("1 " + l1[0] + "\n")	
		else:
			l = line.split(",")
			n = len(l)
			for i in range(n-1):
				outFile.write(l[i] + " ")
			outFile.write(l[n-1])	
outFile.close()