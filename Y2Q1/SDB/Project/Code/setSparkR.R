# Sys.setenv(SPARK_HOME='/home/and/Documents/Projects/Simba/Simba/engine/')
# .libPaths(c(file.path(Sys.getenv('SPARK_HOME'), 'R', 'lib'), .libPaths()))

###
# sparkR --packages com.databricks:spark-csv_2.10:1.5.0
###

library(SparkR)

# sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "2g"))
sc <- sparkR.init("local[4]", "SparkR", sparkPackages="com.databricks:spark-csv_2.10:1.5.0")
sqlContext <- sparkRSQL.init(sc)

schema <- structType(structField("tag", "string"), structField("x", "double"), structField("y", "double"))
points <- read.df(sqlContext, "/home/and/Documents/PhD/Code/Y2Q1/SDB/Project/Code/points.txt", source = "com.databricks.spark.csv", schema = schema)
# points = as.DataFrame(sqlContext = sqlContext, data=points)
registerTempTable(points, "points")
sql = "SELECT * FROM points WHERE POINT(x, y) IN CIRCLERANGE(POINT(4.5, 4.5), 2)"
print(collect(sql(sqlContext,sql)))

