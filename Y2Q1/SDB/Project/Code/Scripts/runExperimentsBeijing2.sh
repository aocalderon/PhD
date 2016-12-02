#!/bin/bash

OUTPUT="Beijing_PBFE_C1_N10K-100K_E10-200"
START=`date`
{ time ./runExperimentsBeijing.sh local > ${OUTPUT}.csv; } 2> .time
TIME=`tail -n 3 .time`
END=`date`
echo -e "${TIME}\nDone!!!"

MSG="Running PBFE with just one core."

echo -e "START:\n${START}\nDATASET:\n${OUTPUT}\nEND:\n${END}\nTIME:\n${TIME}\n\n${MSG}" | mail -s "${OUTPUT} ${START}..." -A ${OUTPUT}.csv acald013@ucr.edu
echo "Email sent."
