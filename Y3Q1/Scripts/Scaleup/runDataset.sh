#!/bin/bash

CORES=$2
PARTITIONS=$3
MU=$4
DSTART=$1
DEND=$1
SUFFIX="K"
ESTART=10.0
EEND=50.0
OUTPUT="Berlin"
TS=`date +%s`
spark-submit ~/PhD/Y2Q4/PFlock/target/scala-2.11/pflock_2.11-1.0.jar \
--prefix "/home/acald013/PhD/Y3Q1/Datasets/B" \
--suffix $SUFFIX \
--master spark://169.235.27.138:7077 \
--mu $MU \
--cores $CORES \
--partitions $PARTITIONS \
--tag $TS \
--estart $ESTART \
--eend $EEND \
--estep 10 \
--dstart $DSTART \
--dend $DEND \
--dstep 20 \
--dirlogs ~/Spark/Logs \
--output $OUTPUT \
--output_md
DATE=`date`
echo "Done!!! $DATE"
