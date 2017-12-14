#!/bin/bash

N=1

# Running Scaleup on 1 Node...
$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh

CORES=7
EPSILON=20
for i in `seq 1 $N`
do
	echo "Running iteration $i/$N for $CORES cores..."
	spark-submit --class RandomTester /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar $EPSILON $CORES
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
	spark-submit --class RandomTester /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar $EPSILON $CORES
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
	spark-submit --class RandomTester /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar $EPSILON $CORES
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
	spark-submit --class RandomTester /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar $EPSILON $CORES
done

$SPARK_HOME/sbin/stop-all.sh
echo "Done!!!"
