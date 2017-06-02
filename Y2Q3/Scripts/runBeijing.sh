#!/bin/bash

spark-submit ~/PhD/Y2Q3/PFlock/target/scala-2.11/pflock_2.11-1.0.jar \
	--prefix /home/acald013/Datasets/Beijing/P \
	--estart 10 --eend 50 --estep 10 \
	--dstart 10 --dend 50 --dstep 10 \
	--partitions 32 \
	--output Beijing 
