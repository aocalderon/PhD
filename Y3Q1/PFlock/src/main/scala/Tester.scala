import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._

object Tester {
	private val log: Logger = LoggerFactory.getLogger("myLogger")

	def main(args: Array[String]): Unit = {
		val spark = SparkSession.builder().
			master("spark://169.235.27.138:7077").
			appName("Tester").
			config("spark.cores.max", "32").
			getOrCreate()
		spark.sparkContext.setLogLevel("INFO")
		import spark.implicits._
		val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
		val path = "Y3Q1/Scripts/Scaleup/"
		val filename1 = "%s%sMaximals_D20K_1S_E10.0_M12_N8_1508217504608.txt".format(phd_home, path)
		val disks1 = spark.sparkContext.textFile(filename1)
		val filename2 = "%s%sMaximals_D20K_2S_E10.0_M12_N356_1508217626817.txt".format(phd_home, path)
		val disks2 = spark.sparkContext.textFile(filename2)
		val filename3 = "%s%sMaximals_D40K_E10.0_M12_N361_1508217759092.txt".format(phd_home, path)
		val disks3 = spark.sparkContext.textFile(filename3)
		val a = disks1.map(d => d.split(",").map(_.toInt).toList.sorted)
		val b = disks2.map(d => d.split(",").map(_.toInt).toList.sorted)
		val c = a.union(b).map(r => (r.mkString(";"), r.length)).toDF("items","clen")
		val d = disks3.map(d => d.split(",").map(_.toInt).toList.sorted).map(r => (r.mkString(";"), r.length)).toDF("items","dlen")
		c.join(d,Seq("items"),"fullouter")
		val r = c.join(d,Seq("items"),"fullouter")
		val c1 = r.filter("dlen IS NULL").select("items","clen") 
		val d1 = r.filter("clen IS NULL").select("items","dlen")
		val cross = c1.repartition(2).crossJoin(d1.repartition(2))
		cross.cache
		val mu = 12
		val f = cross.
			map{ 
				r => (
					r.getString(0), 
					r.getInt(1), 
					r.getString(2), 
					r.getInt(3), 
					r.getString(0).split(";").
						intersect(r.getString(2).split(";")).length
				)		
			}.
			filter(r => r._5 > mu) 
		f.cache
		val n = f.count
		log.info("The count is... %d".format(n))
		spark.close
	}
}
