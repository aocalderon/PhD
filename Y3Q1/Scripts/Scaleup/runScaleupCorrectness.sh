#!/bin/bash

MU=12
PARTITIONS=256

# Running Scaleup on 4 Nodes with different 20K datasets...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack11" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack15" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

./runDatasetCorrectness.sh 1 28 $PARTITIONS $MU
./runDataset.sh 20 28 $PARTITIONS $MU

./runDatasetCorrectness.sh 2 28 $PARTITIONS $MU
./runDataset.sh 40 28 $PARTITIONS $MU

./runDatasetCorrectness.sh 3 28 $PARTITIONS $MU
./runDataset.sh 60 28 $PARTITIONS $MU

./runDatasetCorrectness.sh 4 28 $PARTITIONS $MU
./runDataset.sh 80 28 $PARTITIONS $MU

$SPARK_HOME/sbin/stop-all.sh
echo "Done!!!"
