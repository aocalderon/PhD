#!/bin/bash

START=10
STEP=10
END=20

for dataset in 10K #20K 30K 40K 50K 60K 70K 80K 90K 100K  
do
	for epsilon in `seq $START $STEP $END`
	do
		spark-submit --packages com.databricks:spark-csv_2.10:1.5.0 --master local[*] pbfe.jar /opt/Datasets/Beijing/P${dataset}.csv $epsilon
		cat tdisks/part-* > Beijing_N${dataset}_E${epsilon}.centers
		rm -fR tdisks/
	done
done

