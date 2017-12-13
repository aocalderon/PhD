#!/bin/bash

POINTS=$1
CENTERS=$2
EPSILON=$3
CORES=$4
MASTER=$5
PFLOCK_JAR="/home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar"

spark-submit --class Benchmark $PFLOCK_JAR $POINTS $CENTERS $EPSILON $CORES $MASTER
