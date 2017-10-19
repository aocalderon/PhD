#!/bin/bash

for dataset in 40K 60K 80K
do
	for entries in `seq 2 6 26`
	do
		for partitions in `seq 512 512 3072`
		do
			spark-submit --class Test /home/and/Documents/PhD/Code/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar $dataset $entries $partitions spark://169.235.27.138:7077 28
		done
	done
done
# spark-submit --master spark://169.235.27.138:7077 pbfe3.jar ~/Datasets/Beijing/P${dataset}.csv $epsilon $1
