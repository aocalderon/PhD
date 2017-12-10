#!/bin/bash

POINTS=$1
CENTERS=$2
EPSILON=$3
N=5

# Running Scaleup on 1 Node...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

CORES=7
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N for $CORES cores..."
	./runTester.sh $POINTS $CENTERS $EPSILON $CORES
done

# Running Scaleup on 2 Nodes...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack11" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

CORES=14
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N for $CORES cores..."
	./runTester.sh $POINTS $CENTERS $EPSILON $CORES
done

# Running Scaleup on 3 Nodes...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack11" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

CORES=21
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N for $CORES cores..."
	./runTester.sh $POINTS $CENTERS $EPSILON $CORES
done

# Running Scaleup on 4 Nodes...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack11" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack15" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

CORES=28
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N for $CORES cores..."
	./runTester.sh $POINTS $CENTERS $EPSILON $CORES
done

$SPARK_HOME/sbin/stop-all.sh
echo "Done!!!"
