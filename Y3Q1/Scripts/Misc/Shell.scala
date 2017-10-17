import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.types.StructType
import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.slf4j.Logger
import org.slf4j.LoggerFactory

spark.close

val simba = SimbaSession.builder().
    master("local[*]").
	appName("Shell").
	config("simba.index.partitions", "64").
	config("spark.cores.max", "4").
	getOrCreate()
	
import spark.implicits._
import spark.simbaImplicits._

val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
val path = "Y3Q1/Scripts/Scaleup/"
val filename1 = "%s%sMaximals_D20K_1S_E10.0_M12_N8_1508217504608.txt".format(phd_home, path)
val disks1 = spark.sparkContext.textFile(filename1)
val filename2 = "%s%sMaximals_D20K_2S_E10.0_M12_N356_1508217626817.txt".format(phd_home, path)
val disks2 = spark.sparkContext.textFile(filename2)
val filename3 = "%s%sMaximals_D40K_E10.0_M12_N361_1508217759092.txt".format(phd_home, path)
val disks3 = spark.sparkContext.textFile(filename3)
	
disks1.collect.foreach(println)
disks2.collect.foreach(println)
disks3.collect.foreach(println)

val d1 = disks1.toDF
val d2 = disks2.toDF
val d3 = disks3.toDF

val d = d1.map(d => d.getString(0).split(",").toList).crossJoin(d2.map(d => d.getString(1).split(",").toList))
