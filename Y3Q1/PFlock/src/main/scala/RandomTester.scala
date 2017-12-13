import org.apache.spark.sql.simba.{Dataset, SimbaSession}
import org.apache.spark.sql.simba.index.RTreeType
import scala.collection.mutable.ListBuffer
import org.slf4j.{Logger, LoggerFactory}
import org.scalameter._
import org.apache.spark.sql.functions._
import scala.collection.JavaConverters._

object RandomTester {
  private val logger: Logger = LoggerFactory.getLogger("myLogger")
  val precision = 0.01
  val master = "spark://169.235.27.134:7077"
  val random = scala.util.Random
  var n = 19714
  val nCentersList = List(22482, 47664, 72374, 98838, 125542)
  var epsilon = 0.0
  var cores = 0
  var index = 0
  var timer = new Quantity[Double](0.0, "ms")
  var clock = 0.0

  case class SP_Point(id: Long, x: Double, y: Double)
  case class BBox(minx: Double, miny: Double, maxx: Double, maxy: Double)

  def main(args: Array[String]): Unit = {
    clock = System.nanoTime()
    val simba = SimbaSession.builder().master(master).
      appName("RandomTester").
      //config("simba.join.partitions", "32").
      config("simba.index.partitions", "1024").
      getOrCreate()
    simba.sparkContext.setLogLevel("ERROR")
    logger.info("Starting session...")
    runJoinQuery(simba)
    simba.stop()
  }

  private def runJoinQuery(simba: SimbaSession): Unit = {
    import simba.implicits._
    import simba.simbaImplicits._
    
    epsilon = 50.0
    var nPoints = 0L
    var nCenters = 0L
    
    val minx = 25187
    var miny = 11666
    val maxx = 37625
    var maxy = 20887
    val deltay = maxy - miny + 1000
    var pointsList = new ListBuffer[Dataset[SP_Point]]
    var start = 0
    var end = start + n
    for(index <- (1 to 2)){
      pointsList += (start until end).
        map{ id =>
          var x = minx + random.nextInt((maxx - minx) + 1) + random.nextDouble()
          x = BigDecimal(x).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
          var y = miny + random.nextInt((maxy - miny) + 1) + random.nextDouble()
          y = BigDecimal(y).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
          SP_Point(id, x, y)
        }.toDS()
      start = end
      end = index * n
      miny = index * deltay
      maxy = index * deltay
    }
    var pointsIndex = 0
    val pointsDatasets = new ListBuffer[Dataset[SP_Point]]
    pointsDatasets += pointsList(0).cache()
    pointsDatasets += pointsList(0).union(pointsList(1)).cache()
    //pointsDatasets += pointsList(0).union(pointsList(1).union(pointsList(2))).cache()
    //pointsDatasets += pointsList(0).union(pointsList(1).union(pointsList(2).union(pointsList(3)))).cache()
    
    for(points <- pointsDatasets){
      nPoints = points.count()
      logger.info("Points %d count: %d".format(pointsIndex, nPoints))
      val bboxRow = points.agg(min("x"), min("y"), max("x"), max("y")).collectAsList.asScala.toList
      logger.info("Points %d BBox: %s".format(pointsIndex, BBox2String(getBBox(bboxRow))))
      timer = measure{
        pointsDatasets(pointsIndex) = points.index(RTreeType, "points%dRT".format(pointsIndex), Array("x", "y")).cache()
      }
      logInfo("01.Indexing Points", timer.value, nPoints)
      pointsIndex += 1 
    }
    
////////////////////////////////////////////////////////////////////////

    miny = 11666
    maxy = 20887
    var centersList = new ListBuffer[Dataset[SP_Point]]
    n = 125542
    start = 0
    end = start + n
    for(index <- (1 to 2)){
      centersList += (start until end).
        map{ id =>
          var x = minx + random.nextInt((maxx - minx) + 1) + random.nextDouble()
          x = BigDecimal(x).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
          var y = miny + random.nextInt((maxy - miny) + 1) + random.nextDouble()
          y = BigDecimal(y).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
          SP_Point(id, x, y)
        }.toDS()
      start = end
      end = start + n
      miny += deltay
      maxy += deltay
    }
    var centersIndex = 0
    val centersDatasets = new ListBuffer[Dataset[SP_Point]]
    centersDatasets += centersList(0).cache()
    centersDatasets += centersList(0).union(centersList(1)).cache()
    //centersDatasets += centersList(0).union(centersList(1).union(centersList(2))).cache()
    //centersDatasets += centersList(0).union(centersList(1).union(centersList(2).union(centersList(3)))).cache()
    for(centers <- centersDatasets){
      nCenters = centers.count()
      logger.info("Centers %d count: %d".format(centersIndex, nCenters))
      val bboxRow = centers.agg(min("x"), min("y"), max("x"), max("y")).collectAsList.asScala.toList
      logger.info("Centers %d BBox: %s".format(centersIndex, BBox2String(getBBox(bboxRow))))
      timer = measure{
        centersDatasets(centersIndex) = centers.index(RTreeType, "centers%dRT".format(centersIndex), Array("x", "y")).cache()
      }
      logInfo("02.Indexing Centers", timer.value, nCenters)
      centersIndex += 1 
    }

////////////////////////////////////////////////////////////////////////

    for(index <- (0 until 2)){
      val points = pointsDatasets(index)
      val centers = centersDatasets(index)
      clock = System.nanoTime()
      val disks = centers.
        distanceJoin(points.toDF("id1","x1","y1"), Array("x", "y"), Array("x1", "y1"), epsilon/2 + precision).
        groupBy("id", "x", "y").
        agg(collect_list("id1").alias("ids")).
        cache()
      val nDisks = disks.count()
      logInfo("03.Joining datasets", (System.nanoTime() - clock) / 1e6d, nDisks)
    }

    /*
    timer = measure {
      points = points.index(RTreeType, "pointsRT", Array("x", "y")).cache()
      nPoints = points.count()
    }
    val pointsTimer = timer.value
    epsilon = 10
    for(n <- nCentersList(index)){
      logInfo("01.Indexing points", pointsTimer, nPoints)
      var centers = (0 until n)
        .map {
          id =>
            var x = minx + random.nextInt((maxx - minx) + 1) + random.nextDouble()
            x = BigDecimal(x).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
            var y = miny + random.nextInt((maxy - miny) + 1) + random.nextDouble()
            y = BigDecimal(y).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
            SP_Point(id, x, y)
        }.toDS()
      var nCenters = centers.count()
      timer = measure {
        centers = centers.index(RTreeType, "centersRT", Array("x", "y")).cache()
        nCenters = centers.count()
      }
      logInfo("02.Indexing centers", timer.value, nCenters)
      clock = System.nanoTime()
      val disks = centers.
        distanceJoin(points.toDF("id1","x1","y1"), Array("x", "y"), Array("x1", "y1"), epsilon/2 + precision).
        groupBy("id", "x", "y").
        agg(collect_list("id1").alias("ids")).
        cache()
      val nDisks = disks.count()
      logInfo("03.Joining datasets", (System.nanoTime() - clock) / 1e6d, nDisks)
      epsilon = epsilon + 10
    }
    */
  }
  
  private def getBBox(bboxRow: List[org.apache.spark.sql.Row]): BBox ={
    val row = bboxRow(0)
    BBox(row.getDouble(0), row.getDouble(1), row.getDouble(2), row.getDouble(3))
  }
  
  private def BBox2String(bbox: BBox): String ={
    "(%.2f, %.2f) (%.2f, %.2f)".format(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy)
  }
  
  private def logInfo(msg: String, millis: Double, n: Long): Unit = {
    logger.info("%s,%.2f,%d,%.1f,%d".format(msg, millis / 1000.0, n, epsilon, cores))
  }
}
