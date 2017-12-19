import org.apache.spark.sql.simba.{Dataset, SimbaSession}
import org.apache.spark.sql.simba.index.RTreeType
import org.scalameter._
import org.apache.spark.sql.functions._

object RandomTesterRunner {
  //val logger = LoggerFactory.getLogger(this.getClass)
  //private val logger: Logger = LoggerFactory.getLogger("myLogger")
  
  val precision = 0.01
  var path = ""
  var datasetIndex = 0
  var epsilon = 0.0
  var cores = 0
  var timer = new Quantity[Double](0.0, "ms")
  var clock = 0.0

  case class SP_Point(id: Long, x: Double, y: Double)

  def main(args: Array[String]): Unit = {
    path         = args(0)
    datasetIndex = args(1).toInt
    epsilon      = args(2).toDouble
    cores        = args(3).toInt
    
    val master = "local[%d]".format(cores)
    clock = System.nanoTime()
    val simba = SimbaSession.builder().
      master(master).
      appName("RandomTesterRunner").
      config("simba.index.partitions", "64").
      getOrCreate()
    simba.sparkContext.setLogLevel("ERROR")
    println("Starting session...")
    runJoinQuery(simba)
    simba.stop()
  }

  private def runJoinQuery(simba: SimbaSession): Unit = {
    import simba.implicits._
    import simba.simbaImplicits._
    
    val pointsFilename = "%s/Points%d.txt".format(path, datasetIndex)
    var points = simba.sparkContext.
      textFile(pointsFilename).
      map(_.split(",")).
      map(p => SP_Point(p(0).trim.toLong,p(1).trim.toDouble,p(2).trim.toDouble)).
      toDS()
    var nPoints = points.count()
    timer = measure{
      points = points.index(RTreeType, "points%dRT".format(datasetIndex), Array("x", "y")).cache()
    }
    logInfo("01.Indexing Points", timer.value, nPoints)
    ////////////////////////////////////////////////////////////////////////
    val centersFilename = "%s/Centers%d.txt".format(path, datasetIndex)
    var centers = simba.sparkContext.
      textFile(centersFilename).
      map(_.split(",")).
      map(p => SP_Point(p(0).trim.toLong,p(1).trim.toDouble,p(2).trim.toDouble)).
      toDS()
    var nCenters = centers.count()
    timer = measure{
      centers = centers.index(RTreeType, "centers%dRT".format(datasetIndex), Array("x", "y")).cache()
    }
    logInfo("02.Indexing Centers", timer.value, nCenters)
    ////////////////////////////////////////////////////////////////////////
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
    println("%s,%.2f,%d,%.1f,%d".format(msg, millis / 1000.0, n, epsilon, cores))
  }
}
