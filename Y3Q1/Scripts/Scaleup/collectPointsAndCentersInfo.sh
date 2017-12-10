#!/usr/bin/bash

spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B20K --epsilon 10 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B40K --epsilon 10 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B60K --epsilon 10 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B80K --epsilon 10 --mu 10 --cores 28

spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B20K --epsilon 20 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B40K --epsilon 20 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B60K --epsilon 20 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B80K --epsilon 20 --mu 10 --cores 28

spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B20K --epsilon 30 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B40K --epsilon 30 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B60K --epsilon 30 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B80K --epsilon 30 --mu 10 --cores 28

spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B20K --epsilon 40 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B40K --epsilon 40 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B60K --epsilon 40 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B80K --epsilon 40 --mu 10 --cores 28

spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B20K --epsilon 50 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B40K --epsilon 50 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B60K --epsilon 50 --mu 10 --cores 28
spark-submit --class MaximalFinderExpansion /home/acald013/PhD/Y3Q1/PFlock/target/scala-2.11/pflock_2.11-2.0.jar --dataset B80K --epsilon 50 --mu 10 --cores 28
