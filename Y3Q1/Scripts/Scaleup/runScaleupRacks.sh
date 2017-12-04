#!/bin/bash

DATE=`date`
echo $DATE
echo "[WorkerOrder] 11-12-14-15"
./runScaleupRacks-11-12-14-15.sh
cp nohup.out nohup-11-12-14-15.out
truncate -s 0 nohup.out
DATE=`date`
echo $DATE
printf "\n\n"

DATE=`date`
echo $DATE
echo "[WorkerOrder] 12-11-14-15"
./runScaleupRacks-12-11-14-15.sh
cp nohup.out nohup-12-11-14-15.out
truncate -s 0 nohup.out
DATE=`date`
echo $DATE
printf "\n\n"

DATE=`date`
echo $DATE
echo "[WorkerOrder] 14-12-11-15"
./runScaleupRacks-14-12-11-15.sh
cp nohup.out nohup-14-12-11-15.out
rm nohup.out
DATE=`date`
echo $DATE
printf "\n\n"
