#!/bin/bash

spark-submit ~/PhD/Y2Q3/PFlock/target/scala-2.11/pflock_2.11-1.0.jar \
	--prefix /home/acald013/Datasets/Beijing/P \
	--estart 50 --eend 100 --estep 10 \
	--dstart 50 --dend 100 --dstep 10 \
	--partitions 64 \
	--output Beijing 
