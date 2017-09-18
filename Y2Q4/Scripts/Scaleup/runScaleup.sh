#!/bin/bash

N=3

# Running Scaleup on 1 Node with 20K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack15" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh
for i in `seq 1 $N`
do
	./runDataset.sh 20 7 1024 40
done

# Running Scaleup on 2 Nodes with 40K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack15" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

for i in `seq 1 $N`
do
        ./runDataset.sh 40 14 1024 40
done

# Running Scaleup on 3 Nodes with 60K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack15" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

for i in `seq 1 $N`
do
        ./runDataset.sh 60 21 1024 40
done

# Running Scaleup on 4 Nodes with 80K dataset...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack15" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack11" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

for i in `seq 1 $N`
do
        ./runDataset.sh 80 28 1024 40
done

$SPARK_HOME/sbin/stop-all.sh
echo "Done!!!"
