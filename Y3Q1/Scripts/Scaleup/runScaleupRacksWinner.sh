#!/usr/bin/bash

DATA_PATH="Y3Q1/Datasets/B60Ks/"

DATE=`date`
echo $DATE
echo "[WorkerOrder] 14; 11,12; 11,12,14; 11,12,14,15"
./runScaleupReverseEpsilon.sh $DATA_PATH
cp nohup.out Berlin_14-1112-111214-11121415.out
truncate -s 0 nohup.out
DATE=`date`
echo $DATE
printf "\n\n"


