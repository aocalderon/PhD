from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import Row

conf = (SparkConf()\
	.setMaster("local")\
	.setAppName("My app")\
	.set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

epsilon = 10
points = sc.textFile("P10K.csv")\
	.map(lambda line: line.split(","))\
	.map(lambda p: Row(id=p[0], lat=float(p[1]), lng=float(p[2])))\
	.toDF()

npoints = points.count()
npoints

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
        POINT(p2.lng, p2.lat) IN CIRCLERANGE(POINT(p1.lng, p1.lat), {0}) 
    WHERE 
        p2.id < p1.id""".format(epsilon)

pairs = sqlContext.sql(sql)
print pairs.count()


