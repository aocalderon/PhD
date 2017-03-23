#!/bin/bash

OUTPUT="Beijing_PBFE3_N10K_E10-20"
MSG="Running PBFE3... "
START=`date`
echo -e "$MSG $START\n"
{ time ./runExperimentsBeijing.sh local > ${OUTPUT}.csv; } 2> .time
TIME=`tail -n 3 .time`
END=`date`
echo -e "Done!!! $END\n"
echo -e "$TIME\n"

echo -e "START:\n${START}\nDATASET:\n${OUTPUT}\nEND:\n${END}\nTIME:\n${TIME}\n\n${MSG}" | mail -s "${OUTPUT} ${START}..." -A ${OUTPUT}.csv acald013@ucr.edu
echo -e "Email sent!!!\n"
