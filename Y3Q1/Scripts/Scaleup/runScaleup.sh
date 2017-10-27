#!/bin/bash

N=1
MU=12
PARTITIONS=1024
ESTART=10
EEND=50
ESTEP=10

# Running Scaleup on 1 Node with 20K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack11" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N ..."
	./runDataset.sh B20K $PARTITIONS $ESTART $EEND $ESTEP $MU 7
done

# Running Scaleup on 2 Nodes with 40K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack11" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

for i in `seq 1 $N`
do
	echo "Running iteration $i/$N ..."
	./runDataset.sh B40K $PARTITIONS $ESTART $EEND $ESTEP $MU 14
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
	./runDataset.sh B60K $PARTITIONS $ESTART $EEND $ESTEP $MU 21
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
	./runDataset.sh B80K $PARTITIONS $ESTART $EEND $ESTEP $MU 28
done

$SPARK_HOME/sbin/stop-all.sh
echo "Done!!!"
