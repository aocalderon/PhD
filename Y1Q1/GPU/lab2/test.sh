#!/bin/bash
# declare STRING variable
N=10
for x in `seq 1 $N`; do
	for y in `seq 1 $N`; do
		for z in `seq 1 $N`; do
			./sgemm $x $y $z
		done
	done
done
STRING="Done!!!"
#print variable on a screen
echo $STRING

