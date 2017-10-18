import $ivy.`joda-time:joda-time:2.9.9`
import $ivy.`org.joda:joda-convert:1.8.1`
import $ivy.`org.apache.spark:spark-core_2.11:2.1.0`
import $ivy.`org.apache.spark:spark-catalyst_2.11:2.1.0`
import $ivy.`org.apache.spark:spark-sql_2.11:2.1.0`
import $ivy.`com.vividsolutions:jts-core:1.14.0`

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._
import scala.collection.JavaConverters._

val spark = SparkSession.builder().
    master("local[3]").
	appName("Shell").
	config("spark.cores.max", "3").
	getOrCreate()
	
spark.sparkContext.setLogLevel("ERROR")
import spark.implicits._

val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
val path = "Y3Q1/Scripts/Scaleup/"
val filename1 = "%s%sMaximals_D20K_1S_E10.0_M12_N8_1508217504608.txt".format(phd_home, path)
val disks1 = spark.sparkContext.textFile(filename1)
val filename2 = "%s%sMaximals_D20K_2S_E10.0_M12_N356_1508217626817.txt".format(phd_home, path)
val disks2 = spark.sparkContext.textFile(filename2)
val filename3 = "%s%sMaximals_D40K_E10.0_M12_N361_1508217759092.txt".format(phd_home, path)
val disks3 = spark.sparkContext.textFile(filename3)
	
val a = disks1.map(d => d.split(",").map(_.toInt).toList)
val b = disks2.map(d => d.split(",").map(_.toInt).toList)

val c = a.union(b).map(r => (r, r.length)).toDF("items","clen")
val d = disks3.map(d => d.split(",").map(_.toInt).toList).map(r => (r, r.length)).toDF("items","dlen")

c.join(d,Seq("items"),"fullouter")

val r = c.join(d,Seq("items"),"fullouter")

val c1 = c.join(d,Seq("items"),"fullouter").filter("dlen IS NULL").select("items","clen") 
val d1 = c.join(d,Seq("items"),"fullouter").filter("clen IS NULL").select("items","dlen")

val cross = c1.crossJoin(d1)
cross.cache

val mu = 12
val f = cross.
	map{ 
		r => (
			r.getAs[Seq[Int]](0), 
			r.getInt(1), 
			r.getAs[Seq[Int]](2), 
			r.getInt(3), 
			r.getAs[Seq[Int]](0).
				intersect(r.getAs[Seq[Int]](2)).length
		)
	}.
	filter(r => r._5 > mu) 
f.cache

spark.close

