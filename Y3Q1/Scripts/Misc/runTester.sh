#!/bin/bash

MU=$1
EPSILON=$2
DATASET=$3
P=$4

spark-submit --class MaximalFinder /home/and/Documents/PhD/Code/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar \
--mu $MU \
--estart $EPSILON \
--eend $EPSILON \
--partitions $P \
--entries $P \
--master local[${P}] \
--cores $P \
--dataset $DATASET
