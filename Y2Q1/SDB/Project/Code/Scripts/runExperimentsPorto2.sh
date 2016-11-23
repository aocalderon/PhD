#!/bin/bash

OUTPUT="PortoTest"

./runExperimentsPorto.sh > ${OUTPUT}.csv
echo "Porto 16M, Epsilon 1." | mail -s "Porto experiments..." -A ${OUTPUT}.csv acald013@ucr.edu
