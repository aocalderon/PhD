# coding: utf-8

import os
import sys
from pyproj import Proj, transform

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql import Row

def transformCoords(row):
	inProj = Proj(init='epsg:4326')
	outProj = Proj(init='epsg:4799')
	# print("{0},{1}".format(row.x, row.y))
	try:
		x2,y2 = transform(inProj, outProj, row.x, row.y)
		return Row(x=float(x2), y=float(y2))
	except RuntimeError:
		# print("{0},{1}".format(row.x, row.y))
		return Row(x=float(row.x), y=float(row.y))

def toTag(n):
    if n < 1000:
        return n
    elif n < 1000000:
        return "{0}K".format(int(round(n/1000.0,0)))
    elif n < 1000000000:
        return "{0}M".format(int(round(n/1000000.0,0)))
    else:
        return "{0}B".format(int(round(n/1000000000.0,0)))

sc = SparkContext(appName="SamplingBeijing")
sqlContext = SQLContext(sc)

PATH = sys.argv[1]
TAG = sys.argv[2]
START = int(sys.argv[3])
END = int(sys.argv[4])
STEP = int(sys.argv[5])

schemaIn = StructType([
	StructField("id", IntegerType()),
	StructField("y", DoubleType()),
	StructField("x", DoubleType()),
	StructField("z", DoubleType()),
	StructField("date", StringType())
])

schema = StructType([StructField('X', StringType(), True),StructField('Y', StringType(), True)])

df = sqlContext.read.format('com.databricks.spark.csv').schema(schemaIn).load('file://{0}'.format(PATH))
points = df.select(df.x, df.y).distinct().map(transformCoords)
x = points.count()

print(x)

for i in range(START, END, STEP):
	percentage = float(i) / float(x) 
	sample = points.sample(False, percentage, 42)
	n = sample.count()
	sample.toDF(schema).write.format('com.databricks.spark.csv').option('quote', None).save('output')
	command = "cat output/part-00* > temp.csv"
	os.system(command)
	os.system("rm -fR output/")
	counter = 0
	filename = "{0}{1}.csv".format(TAG, toTag(n))
	outf = open(filename, "w")
	with open("temp.csv", "r") as inf:
		for line in inf:
			fields = line.split(",")
			outf.write("{0},{1},{2}".format(counter, fields[0],fields[1]))
			counter = counter + 1
	os.system("rm -f temp.csv")
	print("{0} has been created...".format(filename))
