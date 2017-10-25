#!/bin/bash

spark-submit --class PartitionSaver /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar B60K 25 1024 40 12 spark://169.235.27.138:7077 28

spark-submit --class PartitionSaver /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar B60K 25 1024 50 12 spark://169.235.27.138:7077 28

