import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index.RTreeType
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType


/**
  * Created by dongx on 3/7/2017.
  */
object BasicSpatialOps {
  case class PointData(x: Double, y: Double, z: Double, other: String)
  case class P1(id1: Long, x1: Double, y1: Double)
  case class P2(id2: Long, x2: Double, y2: Double)

  def main(args: Array[String]): Unit = {
    // "spark://169.235.27.134:7077"
    val simba = SimbaSession.builder().master("local[*]").appName("SparkSessionForSimba").
      config("simba.join.partitions", "7").config("simba.index.partitions", "7").getOrCreate()

    simba.sparkContext.setLogLevel("ERROR")
    runJoinQuery(simba)
    simba.stop()
  }

  private def runJoinQuery(simba: SimbaSession): Unit = {

    import simba.implicits._
    import simba.simbaImplicits._

    val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
    val filename = "%s%s%s.%s".format(phd_home, "Y3Q1/Datasets/", "B20K", "csv")
    val points = simba.read.option("header", "false").text(filename).cache()
    val nPoints = points.count()

    val DS1 = (0 until 20000).map(x => PointData(x, x + 1, x + 2, x.toString)).toDS.index(RTreeType, "ds1RT", Array("x","y"))
    val DS2 = (0 until 20000).map(x => PointData(x, x, x + 1, x.toString)).toDS.index(RTreeType, "ds2RT", Array("x","y"))

    println(DS1.rdd.getNumPartitions)
    println(DS2.rdd.getNumPartitions)

    DS1.distanceJoin(DS2, Array("x", "y"),Array("x", "y"), 3).show()
    
    val p1 = points.map(p => p.getString(0).split(",")).
      map(p => P1(p(0).toLong, p(1).toDouble,p(2).toDouble)).rdd.toDS.
      index(RTreeType,"p1RT",Array("x1","y1"))
    val p2 = points.map(p => p.getString(0).split(",")).map(p => P2(p(0).toLong,p(1).toDouble,p(2).toDouble)).index(RTreeType, "p2RT", Array("x2","y2"))
    println(p1.rdd.getNumPartitions)
    println(p2.rdd.getNumPartitions)
  }
}
