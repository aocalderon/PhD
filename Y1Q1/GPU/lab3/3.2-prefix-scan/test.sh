#!/bin/bash
date
for i in `seq 100000 50000 1000000`; do
	./prefix-scan $i
done
STRING="Done!!!"
echo $STRING
date
