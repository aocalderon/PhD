#!/bin/bash

POINTS=$1
CENTERS=$2
EPSILON=$3
CORES=$4
spark-submit --class BasicSpatialOps /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar $POINTS $CENTERS $EPSILON $CORES
DATE=`date`
echo "Done!!! $DATE"
