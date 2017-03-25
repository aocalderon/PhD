#!/bin/bash

MASTER=$1
START=2
STEP=2
END=10

for dataset in 10K 20K #30K 40K 50K 60K 70K 80K 90K 100K  
do
	for epsilon in `seq $START $STEP $END`
	do
		java -jar PBFE3/pbfe3.jar $MASTER ~/Datasets/Beijing/P${dataset}.csv $epsilon
	done
done

