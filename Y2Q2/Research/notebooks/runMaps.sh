#!/bin/bash
EPSILON=$1
MU=3
ZOOM=15
OUTPUT="test"

./Beijing_Finding_Disks1.R -o $OUTPUT -e $EPSILON -m $MU -z $ZOOM

./captureMap.sh ${OUTPUT}_E${EPSILON}_M${MU}_P1
./captureMap.sh ${OUTPUT}_E${EPSILON}_M${MU}_P2
./captureMap.sh ${OUTPUT}_E${EPSILON}_M${MU}_P3
./captureMap.sh ${OUTPUT}_E${EPSILON}_M${MU}_All

mv -f *_files -t maps/
mv -f *.html -t maps/

scp -i ~/.ssh/dblab *.png acald013@bell.cs.ucr.edu:/home/csgrads/acald013/Images/
mv -f *.png -t figures/
