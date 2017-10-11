#!/bin/bash

EPSILON=50
MU=8

spark-submit /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-1.0.jar \
--estart $EPSILON --eend $EPSILON --estep 10 \
--mstart $MU --mend $MU --mstep 10 \
--tstart 117 --tend 122 \
--master spark://169.235.27.134:7077 \
--cores 28 \
--dataset Berlin_N277K_A18K_T15 \
--partitions 512
