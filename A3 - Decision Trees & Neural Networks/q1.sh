#!/bin/bash

if [ "$#" == 3 ]; then
	echo "Reading input files.."
	python q1_readfiles.py "$1" train
	python q1_readfiles.py "$2" valid
	python q1_readfiles.py "$3" test
	echo "Done. Decision Trees.."
	g++ -std=c++11 -O3 q1.cpp -o dtrees
	./dtrees train valid test
	rm train valid test dtrees
else
	echo "Illegal number of parameters ($#), should be 3"
fi
