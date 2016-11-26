#!/bin/bash

START=2
STEP=1
END=7

for dataset in 1_5M 2M 2_5M 3M 3_5M 4M 
do
	cd /home/and/Documents/PhD/Code/Y2Q1/SDB/Project/Code/Scripts/pbfe2
	./pbfe.job $START $STEP $END /opt/Datasets/Porto/P${dataset}.csv
	cd /home/and/Documents/PhD/Code/Y2Q1/SDB/Project/Code/Scripts/bfe
	./bfe.job $START $STEP $END /opt/Datasets/Porto/P${dataset}.csv
done
