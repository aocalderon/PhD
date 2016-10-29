#!/usr/bin/env python
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row

conf = (SparkConf()
         .setMaster("local")
         .setAppName("My app")
         .set("spark.executor.memory", "2g"))
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

# points = sc.textFile("/home/and/Documents/PhD/Code/Y2Q1/SDB/Project/Code/trajs.csv").map(lambda line: line.split("\t")).map(lambda p: p[1:])
# points = points.toDF(['ts','x','y'])
points = sc.textFile("/home/and/Documents/PhD/Code/Y2Q1/SDB/Project/Code/points.txt").map(lambda line: line.split(",")).map(lambda p: Row(tag=p[0], x=float(p[1]), y=float(p[1])))
points = points.toDF()
points.show()
points.registerTempTable("points")
sql = "SELECT * FROM points WHERE POINT(x, y) IN CIRCLERANGE(POINT(4.5, 4.5), 2)"
sqlContext.sql(sql).show()


# IPYTHON_OPTS="notebook --port 8889 --notebook-dir='/tmp/' --ip='*'" pyspark
