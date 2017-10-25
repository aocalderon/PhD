#!/bin/bash

spark-submit --class PartitionViewer /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar \
B60K 25 64 40 12 spark://169.235.27.138:7077 28

spark-submit --class PartitionViewer /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar \
B60K 25 128 40 12 spark://169.235.27.138:7077 28

spark-submit --class PartitionViewer /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar \
B60K 25 256 40 12 spark://169.235.27.138:7077 28

spark-submit --class PartitionViewer /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar \
B60K 25 512 40 12 spark://169.235.27.138:7077 28

spark-submit --class PartitionViewer /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar \
B60K 25 1024 40 12 spark://169.235.27.138:7077 28

spark-submit --class PartitionViewer /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar \
B60K 25 64 50 12 spark://169.235.27.138:7077 28

spark-submit --class PartitionViewer /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar \
B60K 25 128 50 12 spark://169.235.27.138:7077 28

spark-submit --class PartitionViewer /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar \
B60K 25 256 50 12 spark://169.235.27.138:7077 28

spark-submit --class PartitionViewer /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar \
B60K 25 512 50 12 spark://169.235.27.138:7077 28

spark-submit --class PartitionViewer /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar \
B60K 25 1024 50 12 spark://169.235.27.138:7077 28
