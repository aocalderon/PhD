#!/bin/bash

N=10
DATA_PATH="RandomData/"
EPSILON=100

CORES=2
for i in `seq 1 $N`
do
  echo "Running iteration $i/$N for $CORES cores..."
  spark-submit target/scala-2.11/randomtester_2.11-1.0.jar $DATA_PATH 0 $EPSILON $CORES
done

CORES=4
for i in `seq 1 $N`
do
  echo "Running iteration $i/$N for $CORES cores..."
  spark-submit target/scala-2.11/randomtester_2.11-1.0.jar $DATA_PATH 1 $EPSILON $CORES
done

CORES=6
for i in `seq 1 $N`
do
  echo "Running iteration $i/$N for $CORES cores..."
  spark-submit target/scala-2.11/randomtester_2.11-1.0.jar $DATA_PATH 2 $EPSILON $CORES
done

CORES=8
for i in `seq 1 $N`
do
  echo "Running iteration $i/$N for $CORES cores..."
  spark-submit target/scala-2.11/randomtester_2.11-1.0.jar $DATA_PATH 3 $EPSILON $CORES
done

echo "Done!!!"
