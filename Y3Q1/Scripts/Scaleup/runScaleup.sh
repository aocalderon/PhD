#!/bin/bash

N=5

# Running Scaleup on 1 Node with 20K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

DATASET="B20K"
CORES=7
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N for $DATASET ..."
	./runDataset.sh $DATASET 10 5 $CORES
	./runDataset.sh $DATASET 20 10 $CORES
	./runDataset.sh $DATASET 30 15 $CORES
	./runDataset.sh $DATASET 40 30 $CORES
	./runDataset.sh $DATASET 50 40 $CORES
done

# Running Scaleup on 2 Nodes with 40K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

DATASET="B40K"
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N for $DATASET ..."
	./runDataset.sh B40K 10 5 14
	./runDataset.sh B40K 20 10 14
	./runDataset.sh B40K 30 15 14
	./runDataset.sh B40K 40 30 14
	./runDataset.sh B40K 50 40 14
done

# Running Scaleup on 3 Nodes with 60K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack11" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

DATASET="B60K"
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N for $DATASET ..."
	./runDataset.sh B60K 10 5 21
	./runDataset.sh B60K 20 10 21
	./runDataset.sh B60K 30 15 21
	./runDataset.sh B60K 40 30 21
	./runDataset.sh B60K 50 40 21
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
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N for $DATASET ..."
	./runDataset.sh B80K 10 5 28
	./runDataset.sh B80K 20 10 28
	./runDataset.sh B80K 30 15 28
	./runDataset.sh B80K 40 30 28
	./runDataset.sh B80K 50 40 28
done

$SPARK_HOME/sbin/stop-all.sh
echo "Done!!!"
