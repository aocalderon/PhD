#!/bin/bash

START=1
STEP=1
END=10

for dataset in 1M 2M 4M 
do
	cd /home/and/Documents/PhD/Code/Y2Q1/SDB/Project/Code/Scripts/pbfe2
	./pbfe.job $START $STEP $END /opt/Datasets/Porto/P${dataset}.csv
	cd /home/and/Documents/PhD/Code/Y2Q1/SDB/Project/Code/Scripts/bfe
	./bfe.job $START $STEP $END /opt/Datasets/Porto/P${dataset}.csv
done
