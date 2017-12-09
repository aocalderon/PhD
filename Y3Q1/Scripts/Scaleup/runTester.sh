#!/bin/bash

CORES=$1
spark-submit --class BasicSpatialOps /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar $CORES
DATE=`date`
echo "Done!!! $DATE"
