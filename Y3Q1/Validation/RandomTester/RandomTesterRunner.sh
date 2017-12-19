#!/bin/bash

N=1
DATA_PATH="RandomData/"

CORES=1
for i in `seq 1 $N`
do
  echo "Running iteration $i/$N for $CORES cores..."
  spark-submit target/scala-2.11/randomtester_2.11-1.0.jar $DATA_PATH 0 20 $CORES
done

CORES=2
for i in `seq 1 $N`
do
  echo "Running iteration $i/$N for $CORES cores..."
  spark-submit target/scala-2.11/randomtester_2.11-1.0.jar $DATA_PATH 1 20 $CORES
done

CORES=3
for i in `seq 1 $N`
do
  echo "Running iteration $i/$N for $CORES cores..."
  spark-submit target/scala-2.11/randomtester_2.11-1.0.jar $DATA_PATH 2 20 $CORES
done

CORES=4
for i in `seq 1 $N`
do
  echo "Running iteration $i/$N for $CORES cores..."
  spark-submit target/scala-2.11/randomtester_2.11-1.0.jar $DATA_PATH 3 20 $CORES
done

echo "Done!!!"
