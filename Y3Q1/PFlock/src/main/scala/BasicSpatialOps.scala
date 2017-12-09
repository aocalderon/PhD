import org.apache.spark.sql.simba.{Dataset, SimbaSession}
import org.apache.spark.sql.simba.index.RTreeType
import org.slf4j.{Logger, LoggerFactory}
import org.scalameter._

/**
  * Created by dongx on 3/7/2017.
  */
object BasicSpatialOps {
  private val logger: Logger = LoggerFactory.getLogger("myLogger")
  var clock = 0.0
  case class PointData(x: Double, y: Double, z: Double, other: String)
  case class ST_Point(id: Long, x: Double, y: Double)
  case class P1(id1: Long, x1: Double, y1: Double)
  case class P2(id2: Long, x2: Double, y2: Double)

  def main(args: Array[String]): Unit = {
    //val master = "spark://169.235.27.134:7077"
    clock = System.nanoTime()
    val master = "local[10]"
    val simba = SimbaSession.builder().master("local[4]").
      appName("SparkSessionForSimba").config("simba.join.partitions", "32").
      config("simba.index.partitions", "16").getOrCreate()
    simba.sparkContext.setLogLevel("ERROR")
    logger.info("Starting session,%.2f,%d".format((System.nanoTime() - clock)/1e9d, 0))
    runJoinQuery(simba)
    simba.stop()
  }

  var timer = new Quantity[Double](0.0, "ms")

  private def runJoinQuery(simba: SimbaSession): Unit = {
    clock = System.nanoTime()
    import simba.implicits._
    import simba.simbaImplicits._
    val epsilon = 10
    val minx = 25187
    val maxx = 37625
    val miny = 11666
    val maxy = 20887
    val r = scala.util.Random
    logger.info("Setting variables,%.2f,%d".format((System.nanoTime() - clock)/1e9d, 0))
    clock = System.nanoTime()
    var p1 = (0 until 20000).map {
        id =>
          var x = minx + r.nextInt((maxx - minx) + 1) + r.nextDouble()
          x = BigDecimal(x).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
          var y = miny + r.nextInt((maxy - miny) + 1) + r.nextDouble()
          y = BigDecimal(y).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
          P1(id, x, y)
      }.toDS()
    var p2 = p1.withColumnRenamed("id1", "id2").
        withColumnRenamed("x1", "x2").
        withColumnRenamed("y1", "y2").as[P2]
    logger.info("Generating Data,%.2f,%d".format((System.nanoTime() - clock)/1e9d, 0))
    logger.info("" + p1.rdd.getNumPartitions)
    logger.info("" + p2.rdd.getNumPartitions)
    /*
    clock = System.nanoTime()
    val pairsPreIndex = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon)
    val nPairsPreIndex = pairsPreIndex.count()
    logger.info("DJoin PreIndex,%.2f,%d".format((System.nanoTime() - clock)/1e9d, nPairsPreIndex))
    */
    timer = measure {
      p1 = p1.index(RTreeType, "p1RT", Array("x1", "y1"))
      p2 = p2.index(RTreeType, "p2RT", Array("x2", "y2"))
    }
    logger.info("Indexing,%.2f,%d".format(timer.value / 1000.0, p2.count()))
    logger.info("" + p1.rdd.getNumPartitions)
    logger.info("" + p2.rdd.getNumPartitions)
    clock = System.nanoTime()
    val pairsPostIndex = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon).
      filter(pair => pair.getLong(0) < pair.getLong(3))
    val nPairsPostIndex = pairsPostIndex.count()
    logger.info("DJoin PostIndex,%.2f,%d".format((System.nanoTime() - clock)/1e9d, nPairsPostIndex))
  }
}
