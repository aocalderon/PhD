import org.apache.spark.sql.simba.{Dataset, SimbaSession}
import org.apache.spark.sql.simba.index.RTreeType
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.HashMap
import org.slf4j.{Logger, LoggerFactory}
import org.scalameter._
import org.apache.spark.sql.functions._
import scala.collection.JavaConverters._

object RandomTester {
  private val logger: Logger = LoggerFactory.getLogger("myLogger")
  val precision = 0.01
  val master = "spark://169.235.27.134:7077"
  val random = scala.util.Random
  val nNodes = 4
  var n = 19714
  val nCentersList: HashMap[Double, Int] = HashMap(
    (10.0,22482), 
    (20.0,47664), 
    (30.0, 72374), 
    (40.0, 98838), 
    (50.0, 125542)
  )
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
      config("simba.index.partitions", "1024").
      getOrCreate()
    simba.sparkContext.setLogLevel("ERROR")
    logger.info("Starting session...")
    epsilon = args(0).toDouble
    cores   = args(1).toInt
    runJoinQuery(simba)
    simba.stop()
  }

  private def runJoinQuery(simba: SimbaSession): Unit = {
    import simba.implicits._
    import simba.simbaImplicits._
    
    val minx = 25187
    val miny = 11666
    val maxx = 37625
    val maxy = 20887
    val deltay = maxy - miny + 1000
    val pointsBase = (0 until n).
      map{ id =>
        var x = minx + random.nextInt((maxx - minx) + 1) + random.nextDouble()
        x = BigDecimal(x).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
        var y = miny + random.nextInt((maxy - miny) + 1) + random.nextDouble()
        y = BigDecimal(y).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
        SP_Point(id, x, y)
      }.toDS()
    var nPoints = pointsBase.count()
    var pointsList = new ListBuffer[Dataset[SP_Point]]
    for(index <- (0 until nNodes)){
      pointsList += pointsBase.map{ p: SP_Point => 
        val id = p.id + (index * nPoints)
        val x  = p.x  
        val y  = p.y  + (index * deltay)
        SP_Point(id, x, y)
      }
    }
    for(index <- (1 until nNodes)){
      pointsList(index) = pointsList(index).union(pointsList(index - 1))
    }
    var pointsIndex = 0
    for(points <- pointsList){
      nPoints = points.count()
      //logger.info("Points %d count: %d".format(pointsIndex, nPoints))
      //val bboxRow = points.agg(min("x"), min("y"), max("x"), max("y")).collectAsList.asScala.toList
      //logger.info("Points %d BBox: %s".format(pointsIndex, BBox2String(getBBox(bboxRow))))
      timer = measure{
        pointsList(pointsIndex) = points.index(RTreeType, "points%dRT".format(pointsIndex), Array("x", "y")).cache()
      }
      logInfo("01.Indexing Points", timer.value, nPoints)
      pointsIndex += 1 
    }
    
////////////////////////////////////////////////////////////////////////

    val centersBase = (0 until nCentersList(epsilon)).
      map{ id =>
        var x = minx + random.nextInt((maxx - minx) + 1) + random.nextDouble()
        x = BigDecimal(x).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
        var y = miny + random.nextInt((maxy - miny) + 1) + random.nextDouble()
        y = BigDecimal(y).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
        SP_Point(id, x, y)
      }.toDS()
    var nCenters = centersBase.count()
    var centersList = new ListBuffer[Dataset[SP_Point]]
    for(index <- (0 until nNodes)){
      centersList += centersBase.map{ c: SP_Point => 
        val id = c.id + (index * nCenters)
        val x  = c.x  
        val y  = c.y  + (index * deltay)
        SP_Point(id, x, y)
      }
    }
    for(index <- (1 until nNodes)){
      centersList(index) = centersList(index).union(centersList(index - 1))
    }
    var centersIndex = 0
    for(centers <- centersList){
      nCenters = centers.count()
      //logger.info("Points %d count: %d".format(centersIndex, nCenters))
      //val bboxRow = centers.agg(min("x"), min("y"), max("x"), max("y")).collectAsList.asScala.toList
      //logger.info("Points %d BBox: %s".format(centersIndex, BBox2String(getBBox(bboxRow))))
      timer = measure{
        centersList(centersIndex) = centers.index(RTreeType, "centers%dRT".format(centersIndex), Array("x", "y")).cache()
      }
      logInfo("02.Indexing Centers", timer.value, nCenters)
      centersIndex += 1 
    }

////////////////////////////////////////////////////////////////////////

    for(index <- (0 until nNodes)){
      val points = pointsList(index)
      //saveToFile(points.map(p => "%d,%.2f,%.2f".format(p.id, p.x, p.y)).collect(), "/tmp/Points%d.txt".format(index))
      val centers = centersList(index)
      //saveToFile(centers.map(p => "%d,%.2f,%.2f".format(p.id, p.x, p.y)).collect(), "/tmp/Centers%d.txt".format(index))
      clock = System.nanoTime()
      val disks = centers.
        distanceJoin(points.toDF("id1","x1","y1"), Array("x", "y"), Array("x1", "y1"), epsilon/2 + precision).
        groupBy("id", "x", "y").
        agg(collect_list("id1").alias("ids")).
        cache()
      val nDisks = disks.count()
      logInfo("03.Joining datasets", (System.nanoTime() - clock) / 1e6d, nDisks)
    }
  }
  
  private def saveToFile(data: Array[String], filename: String): Unit = {
    new java.io.PrintWriter(filename) {
      write(data.mkString("\n"))
      close()
    }
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
