library(sparklyr)

Sys.setenv(JAVA_HOME="/usr/lib/jvm/java-8-oracle")
Sys.setenv(SPARK_HOME="/opt/Spark/spark-2.1.0-bin-hadoop2.7")
sc <- spark_connect(master = "local")

trees_tbl <- sdf_copy_to(sc, trees, repartition = 2)
trees_tbl %>%
  spark_apply(function(e) scale(e))