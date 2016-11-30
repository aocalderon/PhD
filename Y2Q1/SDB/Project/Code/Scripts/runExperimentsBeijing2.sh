#!/bin/bash

OUTPUT="Beijing_PBFE2vsPBFE3vsPBFE4_N10K-100K_E10-200"
START=`date`
{ time ./runExperimentsBeijing.sh > ${OUTPUT}.csv; } 2> .time
TIME=`tail -n 3 .time`
END=`date`
echo -e "${TIME}\nDone!!!"

MSG="Comparing PBFE DATAFRAME to RDD vs PBFE DATAFRAME vs PBFE SQL."
echo -e "START:\n${START}\nDATASET:\n${OUTPUT}\nEND:\n${END}\nTIME:\n${TIME}\n\n${MSG}" | mail -s "${OUTPUT} ${START}..." -A ${OUTPUT}.csv acald013@ucr.edu
echo "Email sent."
