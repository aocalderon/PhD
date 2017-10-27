#!/bin/bash

DATASET=$1
PARTITIONS=$2
ESTART=$3
EEND=$4
ESTEP=$5
MU=$6
CORES=$7
ENTRIES=25
MASTER="spark://169.235.27.138:7077"

spark-submit --class MaximalFinder /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar \
--dataset $DATASET \
--estart $ESTART \
--eend $EEND \
--estep $ESTEP \
--mu $MU \
--cores $CORES \
--partitions $PARTITIONS \
--entries $ENTRIES \
--master $MASTER 

DATE=`date`
echo "Done!!! $DATE"
