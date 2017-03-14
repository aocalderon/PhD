#!/bin/bash

START=10
STEP=10
END=200

for dataset in 10K 20K 30K 40K 50K 60K 70K 80K 90K 100K  
do
	for epsilon in `seq $START $STEP $END`
	do
		spark-submit --master local[4] pbfe.jar /opt/Datasets/Beijing/P${dataset}.csv $epsilon
	done
done

