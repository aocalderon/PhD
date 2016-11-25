#!/bin/bash

OUTPUT="Porto1M-4M_E1-10"

./runExperimentsPorto.sh > ${OUTPUT}.csv
echo "Porto 1M 2M 4M, Epsilon 1 to 10." | mail -s "Porto experiments..." -A ${OUTPUT}.csv acald013@ucr.edu
