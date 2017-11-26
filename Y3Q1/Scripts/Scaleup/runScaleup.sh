#!/bin/bash

N=5
CANDIDATES=128

# Running Scaleup on 1 Node with 20K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N ..."
	#./runDataset.sh B20K $CANDIDATES 10 5 7
	#./runDataset.sh B20K $CANDIDATES 20 10 7
	#./runDataset.sh B20K $CANDIDATES 30 15 7
	#./runDataset.sh B20K $CANDIDATES 40 30 7
	#./runDataset.sh B20K $CANDIDATES 50 40 7
done

# Running Scaleup on 2 Nodes with 40K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

for i in `seq 1 $N`
do
	echo "Running iteration $i/$N ..."
	#./runDataset.sh B40K $CANDIDATES 10 5 14
	#./runDataset.sh B40K $CANDIDATES 20 10 14
	#./runDataset.sh B40K $CANDIDATES 30 15 14
	#./runDataset.sh B40K $CANDIDATES 40 30 14
	#./runDataset.sh B40K $CANDIDATES 50 40 14
done

# Running Scaleup on 3 Nodes with 60K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack11" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

for i in `seq 1 $N`
do
	echo "Running iteration $i/$N ..."
	#./runDataset.sh B60K $CANDIDATES 10 5 21
	#./runDataset.sh B60K $CANDIDATES 20 10 21
	#./runDataset.sh B60K $CANDIDATES 30 15 21
	./runDataset.sh B60K $CANDIDATES 40 30 21
	./runDataset.sh B60K $CANDIDATES 50 40 21
done

# Running Scaleup on 4 Nodes with 80K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack11" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack15" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

for i in `seq 1 $N`
do
	echo "Running iteration $i/$N ..."
	#./runDataset.sh B80K $CANDIDATES 10 5 28
	#./runDataset.sh B80K $CANDIDATES 20 10 28
	#./runDataset.sh B80K $CANDIDATES 30 15 28
	./runDataset.sh B80K $CANDIDATES 40 30 28
	./runDataset.sh B80K $CANDIDATES 50 40 28
done

$SPARK_HOME/sbin/stop-all.sh
echo "Done!!!"
