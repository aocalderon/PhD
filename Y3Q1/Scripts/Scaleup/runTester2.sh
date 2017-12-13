#!/bin/bash

POINTS=$1
CENTERS=$2
EPSILON=$3
# Please, modify this variables...
CORES_PER_NODE=7
MASTER="spark://169.235.27.134:7077"
NODE1="169.235.27.137"
NODE2="169.235.27.134"
NODE3="169.235.27.135"
NODE4="169.235.27.138"
N=5

# Running Scaleup on 1 Node...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo $NODE1 >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

CORES=$((CORES_PER_NODE * 1))
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N for $CORES cores..."
	./runTester.sh $POINTS $CENTERS $EPSILON $CORES $MASTER
done

# Running Scaleup on 2 Nodes...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo $NODE2 >> $SPARK_HOME/conf/slaves
echo $NODE3 >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

CORES=$((CORES_PER_NODE * 2))
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N for $CORES cores..."
	./runTester.sh $POINTS $CENTERS $EPSILON $CORES $MASTER
done

# Running Scaleup on 3 Nodes...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo $NODE1 >> $SPARK_HOME/conf/slaves
echo $NODE2 >> $SPARK_HOME/conf/slaves
echo $NODE3 >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

CORES=$((CORES_PER_NODE * 3))
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N for $CORES cores..."
	./runTester.sh $POINTS $CENTERS $EPSILON $CORES $MASTER
done

# Running Scaleup on 4 Nodes...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo $NODE1 >> $SPARK_HOME/conf/slaves
echo $NODE2 >> $SPARK_HOME/conf/slaves
echo $NODE3 >> $SPARK_HOME/conf/slaves
echo $NODE4 >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

CORES=$((CORES_PER_NODE * 4))
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N for $CORES cores..."
	./runTester.sh $POINTS $CENTERS $EPSILON $CORES $MASTER
done

$SPARK_HOME/sbin/stop-all.sh
echo "Done!!!"
