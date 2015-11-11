#!/bin/bash
# declare STRING variable
date
N=10
for i in `seq 1 $1`; do
	x=`shuf -i 100000-1000000 -n 1`
	./reduction $x
done
STRING="Done!!!"
#print variable on a screen
echo $STRING
date
