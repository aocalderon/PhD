#!/bin/bash

N=3
NODES=('acald013@dblab-rack15' 'acald013@dblab-rack14' 'acald013@dblab-rack12' 'acald013@dblab-rack11')
COMBINATIONS=('0' '0 1' '0 1 2' '0 1 2 3')
CORES=('7' '14' '21' '28')
for i in `seq 0 3`
do
	$SPARK_HOME/sbin/stop-all.sh
	truncate -s 0 $SPARK_HOME/conf/slaves
	for j in ${COMBINATIONS[$i]}
	do
		echo "${NODES[$j]}" >> $SPARK_HOME/conf/slaves
	done
	$SPARK_HOME/sbin/start-all.sh
	for k in `seq 1 $N`
	do
		./runBerlin.sh ${CORES[$i]} 1024 50
	done
done
$SPARK_HOME/sbin/stop-all.sh
echo "Done!!!"
