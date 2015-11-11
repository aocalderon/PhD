#!/bin/bash
# declare STRING variable
date
N=10
for i in `seq 1 $1`; do
	x=`shuf -i 100000-1000000 -n 1`
	./prefix-scan $x
done
STRING="Done!!!"
echo $STRING
date
