#!/bin/bash

TS=`date +%s`
DSTART=10
DEND=1
SUFFIX="K"
ESTART=10.0
EEND=10.0
OUTPUT="Beijing"
spark-submit ~/PhD/Y2Q3/PFlock/target/scala-2.11/pflock_2.11-1.0.jar \
	--prefix /home/acald013/Datasets/Beijing/P \
	--estart $ESTART --eend $EEND --estep 2 \
	--dstart $DSTART --dend $DEND --dstep 10 \
	--partitions 16 \
	--output $OUTPUT
FILENAME="${OUTPUT}_N${DSTART}${SUFIX}-${DEND}${SUFFIX}_E${ESTART}-${EEND}_${TS}.csv"
scp -i ~/.ssh/id_rsa $FILENAME acald013@bolt.cs.ucr.edu:/home/csgrads/acald013/public_html/public/Results 
ssh -i ~/.ssh/id_rsa -t acald013@bolt.cs.ucr.edu "plotBenchmarks $FILENAME"
