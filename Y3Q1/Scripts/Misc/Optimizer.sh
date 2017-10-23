#!/bin/bash

EPSILON=50
MU=12
#MASTER="spark://169.235.27.138:7077"
MASTER="local[*]"
CORES=28

$SPARK_HOME/sbin/stop-all.sh
truncate -s 0 $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack11" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack12" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack14" >> $SPARK_HOME/conf/slaves
echo "acald013@dblab-rack15" >> $SPARK_HOME/conf/slaves
$SPARK_HOME/sbin/start-all.sh
`rm /tmp/Andres_Spark*.log`

for dataset in B40K
do
	for entries in `seq 20 6 26`
	do
		for partitions in `seq 512 512 1536`
		do
			spark-submit --class Optimizer /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar \
				$dataset \
				$entries \
				$partitions \
				$EPSILON \
				$MU \
				$MASTER \
				$CORES
		done
	done
done
