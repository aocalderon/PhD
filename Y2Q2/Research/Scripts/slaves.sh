#!/bin/bash

NODES=('acald013@dblab-rack10' 'acald013@dblab-rack11' 'acald013@dblab-rack12' 'acald013@dblab-rack14')
COMBINATIONS=('0' '1' '2' '3' '0 1' '0 2' '0 3' '1 2' '1 3' '2 3' '0 1 2' '0 1 3' '0 2 3' '1 2 3' '0 1 2 3')
for i in `seq 0 14`
do
	truncate -s 0 slaves
	for j in ${COMBINATIONS[$i]}
	do
		echo "${NODES[$j]}" >> slaves
	done
done
