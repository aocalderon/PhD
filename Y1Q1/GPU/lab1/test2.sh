#!/bin/bash
# declare STRING variable
date
N=10
for i in `seq 1000 1000 20000`; do
	./sgemm $i
	./sgemm $i
	./sgemm $i
done
STRING="Done!!!"
#print variable on a screen
echo $STRING
date
