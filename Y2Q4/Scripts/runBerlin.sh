#!/bin/bash

CORES=$1
TS=`date +%s`
PARTITIONS=16
DSTART=160
DEND=160
SUFFIX="K"
ESTART=10.0
EEND=100.0
OUTPUT="Berlin"
echo "Running in $CORES cores..."
spark-submit --files=$SPARK_HOME/conf/metrics.properties ~/PhD/Y2Q4/PFlock/target/scala-2.11/pflock_2.11-1.0.jar \
--prefix /home/acald013/Datasets/Berlin/B \
--master spark://169.235.27.134:7077 \
--cores $CORES \
--partitions $PARTITIONS \
--tag $TS \
--estart $ESTART \
--eend $EEND \
--estep 10 \
--dstart $DSTART \
--dend $DEND \
--dstep 50 \
--dirlogs ~/Spark/Logs \
--output $OUTPUT
#--master spark://169.235.27.134:7077 local[*]\
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
