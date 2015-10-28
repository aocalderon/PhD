#!/bin/bash
# declare STRING variable
date
N=100
for i in `seq 1 $N`; do
	x=`shuf -i 10-1000 -n 1`
	y=`shuf -i 10-1000 -n 1`
	z=`shuf -i 10-1000 -n 1`
	./sgemm-tiled $x $y $z
done
STRING="Done!!!"
#print variable on a screen
echo $STRING
date
