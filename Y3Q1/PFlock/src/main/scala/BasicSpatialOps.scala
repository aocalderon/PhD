import org.apache.spark.sql.simba.{Dataset, SimbaSession}
import org.apache.spark.sql.simba.index.RTreeType
import org.slf4j.{Logger, LoggerFactory}
import org.scalameter._
import org.apache.spark.sql.functions._


/**
  * Created by dongx on 3/7/2017.
  */
object BasicSpatialOps {
  private val logger: Logger = LoggerFactory.getLogger("myLogger")
  var epsilon = 0.0
  val precision = 0.01
  val master = "spark://169.235.27.134:7077"
  var cores = 0
  var pointsFile = ""
  var centersFile = ""
  var timer = new Quantity[Double](0.0, "ms")
  var clock = 0.0

  case class PointData(x: Double, y: Double, z: Double, other: String)
  case class SP_Point(id: Long, x: Double, y: Double)
  case class P1(id1: Long, x1: Double, y1: Double)
  case class P2(id2: Long, x2: Double, y2: Double)

  def main(args: Array[String]): Unit = {
    clock = System.nanoTime()
    pointsFile = args(0)
    centersFile = args(1)
    epsilon = args(2).toDouble
    cores = args(3).toInt
    //master = "local[10]"
    val simba = SimbaSession.builder().master(master).
      appName("Benchmark").
      //config("simba.join.partitions", "32").
      config("simba.index.partitions", "1024").
      getOrCreate()
    simba.sparkContext.setLogLevel("ERROR")
    logger.info("Starting session,%.2f,%d".format((System.nanoTime() - clock)/1e9d, 0))
    runJoinQuery(simba)
    simba.stop()
  }

  private def runJoinQuery(simba: SimbaSession): Unit = {
    clock = System.nanoTime()
    import simba.implicits._
    import simba.simbaImplicits._
    logger.info("Setting variables,%.2f,%d".format((System.nanoTime() - clock)/1e9d, 0))
    clock = System.nanoTime()
    val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
    var path = "Y3Q1/Validation/"
    var dataset = pointsFile
    var extension = "txt"
    var filename = "%s%s%s.%s".format(phd_home, path, dataset, extension)
    var points = simba.sparkContext.
      textFile(filename).
      map { line =>
        val lineArray = line.split(",")
        val id = lineArray(0).toLong
        val x = lineArray(1).toDouble
        val y = lineArray(2).toDouble
        SP_Point(id, x, y)
      }.toDS() //NO CACHE!!!
    var nPoints = points.count()
    path = "Y3Q1/Validation/"
    dataset = centersFile
    extension = "txt"
    filename = "%s%s%s.%s".format(phd_home, path, dataset, extension)
    var centers = simba.sparkContext.
      textFile(filename).
      map { line =>
        val lineArray = line.split(",")
        val id = lineArray(0).toLong
        val x = lineArray(1).toDouble
        val y = lineArray(2).toDouble
        SP_Point(id, x, y)
      }.toDS()
    var nCenters = centers.count()
    logger.info("Reading datasets,%.2f,%d".format((System.nanoTime() - clock)/1e9d, 0))
    logger.info("Points partitions: " + points.rdd.getNumPartitions)
    logger.info("Centers partitions: " + centers.rdd.getNumPartitions)
    timer = measure {
      points = points.index(RTreeType, "pointsRT", Array("x", "y")).cache()
      nPoints = points.count()
    }
    logInfo("01.Indexing points", timer.value, nPoints)
    timer = measure {
      centers = centers.index(RTreeType, "centersRT", Array("x", "y")).cache()
      nCenters = centers.count()
    }
    logInfo("02.Indexing centers", timer.value, nCenters)
    logger.info("" + points.rdd.getNumPartitions)
    logger.info("" + centers.rdd.getNumPartitions)
    clock = System.nanoTime()
    val disks = centers.
      distanceJoin(points.toDF("id1","x1","y1"), Array("x", "y"), Array("x1", "y1"), epsilon/2 + precision).
      groupBy("id", "x", "y").
      agg(collect_list("id1").alias("ids")).
      cache()
    val nDisks = disks.count()
    logInfo("03.Joining datasets", (System.nanoTime() - clock) / 1e6d, nDisks)
  }
  
  private def logInfo(msg: String, millis: Double, n: Long): Unit = {
    logger.info("%s,%.2f,%d,%.1f,%d".format(msg, millis / 1000.0, n, epsilon, cores))
  }
}
