#!/bin/bash

CORES=$1
PARTITIONS=$2
MU=$3
DSTART=80
DEND=80
SUFFIX="K"
ESTART=50.0
EEND=100.0
OUTPUT="Berlin"
TS=`date +%s`
echo "Running in $CORES cores and $PARTITIONS partitions.  Setting mu = $MU ..."
spark-submit ~/PhD/Y2Q4/PFlock/target/scala-2.11/pflock_2.11-1.0.jar \
--prefix /home/acald013/Datasets/Berlin/EPSG3068/B \
--suffix $SUFFIX \
--master spark://169.235.27.138:7077 \
--mu $MU \
--cores $CORES \
--partitions $PARTITIONS \
--tag $TS \
--estart $ESTART \
--eend $EEND \
--estep 10 \
--dstart $DSTART \
--dend $DEND \
--dstep 20 \
--dirlogs ~/Spark/Logs \
--output $OUTPUT
# --master spark://169.235.27.134:7077 local[*] dblab-rack11=169.235.27.134 dblab-rack15=169.235.27.138
# --files=$SPARK_HOME/conf/metrics.properties
#TS2=`date +%s`
#DELAY=printf %.2f $(echo "($TS2-$TS1)/60" | bc -l)
#echo "Done at ... ${DELAY}s"
#FILENAME="${OUTPUT}_N${DSTART}${SUFFIX}-${DEND}${SUFFIX}_E${ESTART}-${EEND}_C${CORES}_${TS}.csv"
#scp -i ~/.ssh/id_rsa $FILENAME acald013@bolt.cs.ucr.edu:/home/csgrads/acald013/public_html/public/Results 
#ssh -i ~/.ssh/id_rsa -t acald013@bolt.cs.ucr.edu "plotBenchmarks $FILENAME"
#cd ~/PhD/
#DATE=`date`
#git add --all
#git commit -m "Adding plots for $FILENAME on $DATE ..."
#git pull
#git push
#cd ~/
echo "Done!!!"
