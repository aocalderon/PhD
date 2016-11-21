#!/bin/bash

for dataset in `seq 10 10 100`
do
	cd /home/and/Documents/PhD/Code/Y2Q1/SDB/Project/Code/Scripts/pbfe2
	./pbfe.job 10 10 200 P${dataset}K.csv
	cd /home/and/Documents/PhD/Code/Y2Q1/SDB/Project/Code/Scripts/bfe
	./bfe.job 10 10 200 P${dataset}K.csv
done
