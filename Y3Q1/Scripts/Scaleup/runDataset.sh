#!/bin/bash

DATASET=$1
CANDIDATES=$2
EPSILON=$3
MU=$4
CORES=$5

spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar \
--dataset $DATASET \
--epsilon $EPSILON \
--mu $MU \
--cores $CORES 

DATE=`date`
echo "Done!!! $DATE"
