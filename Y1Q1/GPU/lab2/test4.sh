#!/bin/bash
# declare STRING variable
date
N=10
for i in `seq 1000 1000 20000`; do
	./sgemm-tiled $i
	./sgemm-tiled $i
	./sgemm-tiled $i
done
STRING="Done!!!"
#print variable on a screen
echo $STRING
date
