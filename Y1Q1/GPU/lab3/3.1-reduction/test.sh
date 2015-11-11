#!/bin/bash
date
for i in `seq 800000 10000 1000000`; do
	./reduction $i
done
STRING="Done!!!"
echo $STRING
date
