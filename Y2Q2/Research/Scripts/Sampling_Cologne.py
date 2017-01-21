# coding: utf-8

import os
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql import Row

def toTag(n):
    if n < 1000:
        return n
    elif n < 1000000:
        return "{0}K".format(int(round(n/1000.0,0)))
    elif n < 1000000000:
        return "{0}M".format(int(round(n/1000000.0,0)))
    else:
        return "{0}B".format(int(round(n/1000000000.0,0)))

sc = SparkContext(appName="Sampling")
sqlContext = SQLContext(sc)

df = sqlContext.read.format('com.databricks.spark.csv').load('file:///home/and/Documents/PhD/Code/Y2Q2/Research/Scripts/C.csv')
df = df.sample(False,0.01,42)
df.take(5)

points = df.map(lambda point: Row(point[1],point[2])).distinct()
points.cache()

points.count()

schema = StructType([StructField('X', StringType(), True),StructField('Y', StringType(), True)])
x = 731016

percentage = x / x
sample = points.sample(False, percentage, 42)
n = sample.count()

sample.toDF(schema).write.format('com.databricks.spark.csv').option('quote', None).save('output')
print(n)

command = "cat output/part-00* > L{0}.csv".format(toTag(n))
os.system(command)
os.system("rm -fR output/")



