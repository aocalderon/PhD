#!/bin/bash

N=3
MU=12
PARTITIONS=512

# Running Scaleup on 4 Nodes with different 20K datasets...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack11" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack15" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

for i in `seq 1 $N`
do
	echo "Running iteration $i ..."
        ./runDatasetCorrectness.sh 1 28 $PARTITIONS $MU
done

for i in `seq 1 $N`
do
	echo "Running iteration $i ..."
        ./runDatasetCorrectness.sh 2 28 $PARTITIONS $MU
done

for i in `seq 1 $N`
do
	echo "Running iteration $i ..."
        ./runDatasetCorrectness.sh 3 28 $PARTITIONS $MU
done

for i in `seq 1 $N`
do
	echo "Running iteration $i ..."
        ./runDatasetCorrectness.sh 4 28 $PARTITIONS $MU
done

$SPARK_HOME/sbin/stop-all.sh
echo "Done!!!"
