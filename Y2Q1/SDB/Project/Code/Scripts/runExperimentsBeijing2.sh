#!/bin/bash

OUTPUT="Beijing_PBFE2_N10K-100K_E10-200"
START=`date`
./runExperimentsBeijing.sh > ${OUTPUT}.csv
END=`date`
echo -e "START:\n${START}\nDATASET:\n${OUTPUT}\nEND:\n${END}\n\nJust PBFE2" | mail -s "${OUTPUT} ${START}..." -A ${OUTPUT}.csv acald013@ucr.edu
