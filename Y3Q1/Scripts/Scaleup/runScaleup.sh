#!/bin/bash

N=5
EPSILONS=(30 40 50 60)
MUS=(25 25 25 25)
M=${#EPSILONS[@]}

# Running Scaleup on 1 Node with 20K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

DATASET="B20K"
CORES=7
for i in `seq 1 $N`
do
	for(( j=0; j<${M}; j++ ));
	do 
		echo "Running iteration $i/$N for $DATASET (epsilon = ${EPSILONS[$j]} , mu = ${MUS[$j]})..."
		./runDataset.sh $DATASET ${EPSILONS[$j]} ${MUS[$j]} $CORES
	done
done

# Running Scaleup on 2 Nodes with 40K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

DATASET="B40K"
CORES=14
for i in `seq 1 $N`
do
	for(( j=0; j<${M}; j++ ));
	do 
		echo "Running iteration $i/$N for $DATASET (epsilon = ${EPSILONS[$j]} , mu = ${MUS[$j]})..."
		./runDataset.sh $DATASET ${EPSILONS[$j]} ${MUS[$j]} $CORES
	done
done

# Running Scaleup on 3 Nodes with 60K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack11" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

DATASET="B60K"
CORES=21
for i in `seq 1 $N`
do
	for(( j=0; j<${M}; j++ ));
	do 
		echo "Running iteration $i/$N for $DATASET (epsilon = ${EPSILONS[$j]} , mu = ${MUS[$j]})..."
		./runDataset.sh $DATASET ${EPSILONS[$j]} ${MUS[$j]} $CORES
	done
done

# Running Scaleup on 4 Nodes with 80K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack11" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack15" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

DATASET="B80K"
CORES=28
for i in `seq 1 $N`
do
	for(( j=0; j<${M}; j++ ));
	do 
		echo "Running iteration $i/$N for $DATASET (epsilon = ${EPSILONS[$j]} , mu = ${MUS[$j]})..."
		./runDataset.sh $DATASET ${EPSILONS[$j]} ${MUS[$j]} $CORES
	done
done

$SPARK_HOME/sbin/stop-all.sh
echo "Done!!!"
