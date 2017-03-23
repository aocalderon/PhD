#!/bin/bash

START=10
STEP=10
END=20

for dataset in 10K #20K 30K 40K 50K #60K 70K 80K 90K 100K  
do
	for epsilon in `seq $START $STEP $END`
	do
		spark-submit --master local[*] target/scala-2.10/pbfe3_2.10-1.0.jar /opt/Datasets/Beijing/P${dataset}.csv $epsilon
	done
done

