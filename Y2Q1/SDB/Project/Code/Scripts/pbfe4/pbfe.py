#!/usr/bin/env python3
import math
import time
import sys
from os.path import basename, splitext
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row

conf = (SparkConf().setMaster("local[*]").setAppName("PBFE"))
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

filename = sys.argv[1]
epsilon = float(sys.argv[2])
r2 = math.pow(epsilon/2,2)
tag = splitext(basename(filename))[0][1:]

def calculateDisks(pair):
    global epsilon
    global r2
    
    X = pair[1] - pair[4]
    Y = pair[2] - pair[5]
    D2 = math.pow(X, 2) + math.pow(Y, 2)
    if (D2 == 0):
        return []
    expression = abs(4 * (r2 / D2) - 1)
    root = math.pow(expression, 0.5)
    h1 = ((X + Y * root) / 2) + pair[4]
    h2 = ((X - Y * root) / 2) + pair[4]
    k1 = ((Y - X * root) / 2) + pair[5]
    k2 = ((Y + X * root) / 2) + pair[5]
    
    return Row(id1=pair[0],id2=pair[3],lat1=h1,lng1=k1,lat2=h2,lng2=k2)

points = sc.textFile(filename).map(lambda line: line.split(",")).map(lambda p: Row(id=p[0], lat=float(p[1]), lng=float(p[2]))).toDF()
t1 = time.time()
points.registerTempTable("p1")
points.registerTempTable("p2")
sql = """
    SELECT 
        * 
    FROM 
        p1 
    DISTANCE JOIN 
        p2 
    ON 
        POINT(p2.lng, p2.lat) IN CIRCLERANGE(POINT(p1.lng, p1.lat), {0})""".format(epsilon)
pairs = sqlContext.sql(sql).rdd.filter(lambda pair: pair[0] < pair[3]).map(calculateDisks)
ndisks = pairs.count()
t2 = round(time.time() - t1,3)
print("PBFE-SQL-POSTFILTER,{0},{1},{2},{3}".format(float(epsilon),tag,2 * ndisks,t2))



