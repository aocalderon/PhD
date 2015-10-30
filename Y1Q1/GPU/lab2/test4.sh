#!/bin/bash
date
for i in `seq 50 50 1000`; do
	./sgemm-tiled $i
	./sgemm-tiled $i
	./sgemm-tiled $i
done
STRING="Done!!!"
echo $STRING
date
