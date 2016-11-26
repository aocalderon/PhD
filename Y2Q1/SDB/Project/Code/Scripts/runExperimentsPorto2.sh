#!/bin/bash

OUTPUT="Porto_N1.5M-4M_E2-7"

./runExperimentsPorto.sh > ${OUTPUT}.csv
echo "Porto 1.5M to 4M, Epsilon 2 to 7." | mail -s "Porto experiments..." -A ${OUTPUT}.csv acald013@ucr.edu
