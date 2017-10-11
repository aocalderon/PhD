#!/bin/bash

spark-submit /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-1.0.jar --estart 50 --eend 100 --estep 10 --mstart 10 --mend 40 --mstep 10 --tend 118 --master spark://169.235.27.134:7077 --cores 28 --dataset Berlin_N277K_A18K_T15
