#!/bin/bash

#This is simple example how to use CTL for training and testing.

#The train set is a very small part of training set with 1,000 documents, and 200 documents are for testing.
#
#Check ../demo/ to show the input files: training set, test set.
#Check ./output to show the output 

make clean
echo
make
echo
rm -f ./output/*

echo

time ./ratm est ./input/train setting.txt 10 4 N ./output ./input/init_beta

echo

time ./ctl inf ./demo/test_wiki200.txt setting.txt ./output final ./output
