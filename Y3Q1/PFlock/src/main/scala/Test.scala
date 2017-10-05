import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.types.StructType
import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.apache.spark.sql.simba.index.RTreeType

case class ST_Point(x: Double, y: Double, t: Int, id: Int)

val POINT_SCHEMA = ScalaReflection.schemaFor[ST_Point].dataType.asInstanceOf[StructType]
val simba = SimbaSession.builder().master("local[3]").appName("Runner").config("simba.index.partitions", "64").config("spark.cores.max", "3").getOrCreate()
import simba.implicits._
import simba.simbaImplicits._
val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
val filename = s"${phd_home}Y3Q1/Datasets/Berlin_N15K_A1K_T15.csv"
val dataset = simba.read.option("header", "false").schema(POINT_SCHEMA).csv(filename).as[ST_Point].filter(datapoint => datapoint.t < 120)
val d = dataset.index(RTreeType, "dRT", Array("x", "y"))
val c = d.count()
d.cache
val timestamps = d.map(datapoint => datapoint.t).distinct.sort("value").collect.toList
var timestamp = timestamps.head
var currentPoints = d.filter(datapoint => datapoint.t == timestamp).map(datapoint => PFlock.SP_Point(datapoint.id, datapoint.x, datapoint.y))
//var cp = currentPoints.index(RTreeType, "cpRT", Array("x", "y"))
PFlock.EPSILON = 100.0
PFlock.MU = 3
val f0: RDD[List[Int]] = PFlock.run(currentPoints, timestamp, simba)
//currentPoints.dropIndexByName("cpRT")

timestamp = timestamps(1)
currentPoints = d.filter(datapoint => datapoint.t == timestamp).map(datapoint => PFlock.SP_Point(datapoint.id, datapoint.x, datapoint.y))
//cp = currentPoints.index(RTreeType, "cpRT", Array("x", "y"))
val f1: RDD[List[Int]] = PFlock.run(currentPoints, timestamp, simba)
//currentPoints.dropIndexByName("cpRT")

f0.cartesian(f1).foreach(println)

