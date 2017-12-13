import org.apache.spark.sql.simba.{Dataset, SimbaSession}
import org.apache.spark.sql.simba.index.RTreeType
import org.slf4j.{Logger, LoggerFactory}
import org.scalameter._
import org.apache.spark.sql.functions._

object Benchmark {
  private val logger: Logger = LoggerFactory.getLogger("myLogger")
  val precision = 0.01
  var master = ""
  var epsilon = 0.0
  var cores = 0
  var pointsFile = ""
  var centersFile = ""
  var timer = new Quantity[Double](0.0, "ms")
  var clock = 0.0

  case class SP_Point(id: Long, x: Double, y: Double)

  def main(args: Array[String]): Unit = {
    clock = System.nanoTime()
    pointsFile = args(0)
    centersFile = args(1)
    epsilon = args(2).toDouble
    cores = args(3).toInt
    master = args(4)
    val simba = SimbaSession.builder().master(master).
      appName("Benchmark").
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
    val path = scala.util.Properties.envOrElse("DATA_HOME", "/home/acald013/PhD/")
    var dataset = pointsFile
    var extension = "txt"
    var filename = "%s%s.%s".format(path, dataset, extension)
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
    dataset = centersFile
    extension = "txt"
    filename = "%s%s.%s".format(path, dataset, extension)
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
    logger.info("Reading datasets... %.2fs".format((System.nanoTime() - clock)/1e9d))
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
    clock = System.nanoTime()
    val disks = points.
      distanceJoin(centers.toDF("id1","x1","y1"), Array("x", "y"), Array("x1", "y1"), epsilon/2 + precision).
      groupBy("id1", "x1", "y1").
      agg(collect_list("id").alias("ids")).
      cache()
    val nDisks = disks.count()
    logInfo("03.Joining datasets", (System.nanoTime() - clock) / 1e6d, nDisks)
  }
  
  private def logInfo(msg: String, millis: Double, n: Long): Unit = {
    logger.info("%s,%.2f,%d,%.1f,%d".format(msg, millis / 1000.0, n, epsilon, cores))
  }
}
