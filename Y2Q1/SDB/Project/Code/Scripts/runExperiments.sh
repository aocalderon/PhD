#!/bin/bash

for dataset in `seq 10 10 30`
do
	cd /home/and/Documents/PhD/Code/Y2Q1/SDB/Project/Code/Scripts/pbfe2
	./pbfe.job 20 20 100 P${dataset}K.csv
	cd /home/and/Documents/PhD/Code/Y2Q1/SDB/Project/Code/Scripts/bfe
	./bfe.job 20 20 100 P${dataset}K.csv
done
