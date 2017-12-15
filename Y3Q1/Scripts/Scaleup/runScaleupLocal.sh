#!/bin/bash

N=5
EPSILONS=(10 20 30 40 50)
MUS=(10 10 10 10 10)
#EPSILONS=(10)
#MUS=(10)
M=${#EPSILONS[@]}

$SPARK_HOME/sbin/stop-all.sh

# Running Scaleup on 1 Node...
DATA_PATH="Y3Q1/Datasets/B60Ks/"
DATASET="B60K"
CORES=2
MASTER="local[2]"
for i in `seq 1 $N`
do
	for(( j=0; j<${M}; j++ ));
	do 
		echo "Running iteration $i/$N for $DATASET (cores = $CORES, epsilon = ${EPSILONS[$j]} , mu = ${MUS[$j]})..."
		./runDatasetLocal.sh $DATA_PATH $DATASET ${EPSILONS[$j]} ${MUS[$j]} $CORES $MASTER
	done
done

# Running Scaleup on 2 Nodes...
DATASET="B120K"
CORES=4
MASTER="local[4]"
for i in `seq 1 $N`
do
	for(( j=0; j<${M}; j++ ));
	do 
		echo "Running iteration $i/$N for $DATASET (cores = $CORES, epsilon = ${EPSILONS[$j]} , mu = ${MUS[$j]})..."
		./runDatasetLocal.sh $DATA_PATH $DATASET ${EPSILONS[$j]} ${MUS[$j]} $CORES $MASTER
	done
done

# Running Scaleup on 3 Nodes...
DATASET="B180K"
CORES=21
MASTER="local[6]"
for i in `seq 1 $N`
do
	for(( j=0; j<${M}; j++ ));
	do 
		echo "Running iteration $i/$N for $DATASET (cores = $CORES, epsilon = ${EPSILONS[$j]} , mu = ${MUS[$j]})..."
		./runDatasetLocal.sh $DATA_PATH $DATASET ${EPSILONS[$j]} ${MUS[$j]} $CORES $MASTER
	done
done

# Running Scaleup on 4 Nodes...
DATASET="B240K"
CORES=28
MASTER="local[8]"
for i in `seq 1 $N`
do
	for(( j=0; j<${M}; j++ ));
	do 
		echo "Running iteration $i/$N for $DATASET (cores = $CORES, epsilon = ${EPSILONS[$j]} , mu = ${MUS[$j]})..."
		./runDatasetLocal.sh $DATA_PATH $DATASET ${EPSILONS[$j]} ${MUS[$j]} $CORES $MASTER
	done
done

echo "Done!!!"
