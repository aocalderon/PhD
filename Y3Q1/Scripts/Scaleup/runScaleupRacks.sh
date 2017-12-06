#!/bin/bash

DATE=`date`
echo $DATE
echo "[WorkerOrder] 14; 11,12; 11,12,14; 11,12,14,15"
./runScaleupRacks_14-1112-111214-11121415.sh
cp nohup.out Berlin_14-1112-111214-11121415.out
truncate -s 0 nohup.out
DATE=`date`
echo $DATE
printf "\n\n"

DATE=`date`
echo $DATE
echo "[WorkerOrder] 14; 11,14; 11,12,14; 11,12,14,15"
./runScaleupRacks_14-1114-111214-11121415.sh
cp nohup.out Berlin_14-1114-111214-11121415.out
truncate -s 0 nohup.out
DATE=`date`
echo $DATE
printf "\n\n"

DATE=`date`
echo $DATE
echo "[WorkerOrder] 14; 12,14; 11,12,14; 11,12,14,15"
./runScaleupRacks_14-1214-111214-11121415.sh
cp nohup.out Berlin_14-1214-111214-11121415.out
rm nohup.out
DATE=`date`
echo $DATE
printf "\n\n"

##

DATE=`date`
echo $DATE
echo "[WorkerOrder] 12; 11,14; 11,12,14; 11,12,14,15"
./runScaleupRacks_14-1114-111214-11121415.sh
cp nohup.out Berlin_14-1114-111214-11121415.out
rm nohup.out
DATE=`date`
echo $DATE
printf "\n\n"

DATE=`date`
echo $DATE
echo "[WorkerOrder] 12; 11,12; 11,12,14; 11,12,14,15"
./runScaleupRacks_14-1114-111214-11121415.sh
cp nohup.out Berlin-12-11-14-15.out
truncate -s 0 nohup.out
DATE=`date`
echo $DATE
printf "\n\n"

DATE=`date`
echo $DATE
echo "[WorkerOrder] 12; 12,14; 11,12,14; 11-12-14-15"
./runScaleupRacks-11-12-14-15.sh
cp nohup.out Berlin-11-12-14-15.out
truncate -s 0 nohup.out
DATE=`date`
echo $DATE
printf "\n\n"

##

DATE=`date`
echo $DATE
echo "[WorkerOrder] 11; 12,14; 11,12,14; 11-12-14-15"
./runScaleupRacks-11-12-14-15.sh
cp nohup.out Berlin-11-12-14-15.out
truncate -s 0 nohup.out
DATE=`date`
echo $DATE
printf "\n\n"

DATE=`date`
echo $DATE
echo "[WorkerOrder] 11; 11,12; 11,12,14; 11-12-14-15"
./runScaleupRacks-11-12-14-15.sh
cp nohup.out Berlin-11-12-14-15.out
truncate -s 0 nohup.out
DATE=`date`
echo $DATE
printf "\n\n"

DATE=`date`
echo $DATE
echo "[WorkerOrder] 11; 11,14; 11,12,14; 11-12-14-15"
./runScaleupRacks-11-12-14-15.sh
cp nohup.out Berlin-11-12-14-15.out
truncate -s 0 nohup.out
DATE=`date`
echo $DATE
printf "\n\n"


