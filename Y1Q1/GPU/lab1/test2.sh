#!/bin/bash
# declare STRING variable
date
N=10
for i in `seq 50 50 1000`; do
	./sgemm $i
	./sgemm $i
	./sgemm $i
done
STRING="Done!!!"
#print variable on a screen
echo $STRING
date
