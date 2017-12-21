#!/bin/bash

PARTITIONS=1024
EPSILONS=(10 20 30 40 50)
DATASETS=(B20K B40K B60K B80K)
M=${#EPSILONS[@]}
N=${#DATASETS[@]}

for(( i=0; i<${N}; i++ ));
do
	for(( j=0; j<${M}; j++ ));
	do 
		echo "Running iteration dataset = ${DATASETS[$i]} and epsilon = ${EPSILONS[$j]}..."
    spark-submit /home/acald013/PhD/Y3Q1/Validation/PartitionViewer/target/scala-2.11/partitionviewer_2.11-1.0.jar --dataset ${DATASETS[$i]} --epsilon ${EPSILONS[$j]} --partitions $PARTITIONS 
	done
done

