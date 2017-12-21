import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index.{RTree, RTreeType}
import org.apache.spark.sql.simba.partitioner.STRPartitioner
import org.apache.spark.sql.simba.spatial.{MBR, Point}
import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.slf4j.{Logger, LoggerFactory}

object PartitionViewer {
	private val logger: Logger = LoggerFactory.getLogger("myLogger")
  private val precision: Double = 0.01
  private val dimensions: Int = 2
  private val sampleRate: Double = 0.01
  private var phd_home = ""
  private var nPoints = 0L

  case class SP_Point(id: Long, x: Double, y: Double)
  case class P1(id1: Long, x1: Double, y1: Double)
  case class P2(id2: Long, x2: Double, y2: Double)
  case class Center(id: Long, x: Double, y: Double)
  case class Pair(id: Long, x: Double, y: Double, id2: Long, x2: Double, y2: Double)

  def run(pointsRDD: RDD[String]
      , simba: SimbaSession
      , conf: Conf): Unit = {
    // 00.Setting variables...
    val epsilon = conf.epsilon()
    import simba.implicits._
    import simba.simbaImplicits._
    val startTime = System.currentTimeMillis()
    // 01.Indexing points...
    var timer = System.currentTimeMillis()
    var pointsNumPartitions = pointsRDD.getNumPartitions
    logger.info("[Partitions Info]Points;Before indexing;%d".format(pointsNumPartitions))
    val p1 = pointsRDD.map(_.split(",")).
      map(p => SP_Point(p(0).trim.toLong,p(1).trim.toDouble,p(2).trim.toDouble)).
      toDS().
      index(RTreeType,"p1RT",Array("x","y")).
      cache()
    val p2 = pointsRDD.map(_.split(",")).
      map(p => P2(p(0).trim.toLong,p(1).trim.toDouble,p(2).trim.toDouble)).
      toDS().
      index(RTreeType,"p2RT",Array("x2","y2")).
      cache()
    nPoints = p1.count()
    pointsNumPartitions = p1.rdd.getNumPartitions
    logger.info("[Partitions Info]Points;After indexing;%d".format(pointsNumPartitions))
    logger.info("01.Indexing points... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nPoints))
    ///////////
          val pointPoints = p1.map{ point =>
              ( new Point(Array(point.x, point.y)), point)
            }
            .rdd
            .cache()
          val pointsSampleRate: Double = sampleRate
          val pointsDimension: Int = dimensions
          val pointsTransferThreshold: Long = 800 * 1024 * 1024
          val pointsMaxEntriesPerNode: Int = 25
          val pointsPartitioner: STRPartitioner = new STRPartitioner(
            pointsNumPartitions,
            pointsSampleRate,
            pointsDimension,
            pointsTransferThreshold,
            pointsMaxEntriesPerNode,
            pointPoints )
          val pointsMBRs = pointsPartitioner.mbrBound.map{ mbr =>
              "%d;%s".format(mbr._2, mbr2wkt(mbr._1))
            }
          saveStringArrayWithoutTimeMillis(pointsMBRs, "PointsMBRs", conf)
    ///////////
    // 02.Getting pairs...
    timer = System.currentTimeMillis()
    val pairs = p1.distanceJoin(p2, Array("x", "y"), Array("x2", "y2"), epsilon + precision)
      .as[Pair]
      .filter(pair => pair.id < pair.id2)
      .rdd
      .cache()
    val nPairs = pairs.count()
    logger.info("02.Getting pairs... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nPairs))
    // 03.Computing centers...
    timer = System.currentTimeMillis()
    val centerPairs = findCenters(pairs, epsilon)
      .filter( pair => pair.id != -1 )
      .toDS()
      .as[Pair]
      .cache()
    val leftCenters = centerPairs.select("x","y")
    val rightCenters = centerPairs.select("x2","y2")
    val centersRDD = leftCenters.union(rightCenters)
      .toDF("x", "y")
      .withColumn("id", monotonically_increasing_id())
      .as[SP_Point]
      .rdd
      .repartition(conf.cores())
      .cache()
    val nCenters = centersRDD.count()
    logger.info("03.Computing centers... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCenters))
    // 04.Indexing centers...
    timer = System.currentTimeMillis()
    var centersNumPartitions: Int = centersRDD.getNumPartitions
    logger.info("[Partitions Info]Centers;Before indexing;%d".format(centersNumPartitions))
    val centers = centersRDD.toDS.index(RTreeType, "centersRT", Array("x", "y")).cache()
    centersNumPartitions = centers.rdd.getNumPartitions
    logger.info("[Partitions Info]Centers;After indexing;%d".format(centersNumPartitions))
    logger.info("04.Indexing centers... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCenters))
    ///////////
          val pointCenters = centers.map{ center =>
              ( new Point(Array(center.x, center.y)), center)
            }
            .rdd
            .cache()
          val centersSampleRate: Double = sampleRate
          val centersDimension: Int = dimensions
          val centersTransferThreshold: Long = 800 * 1024 * 1024
          val centersMaxEntriesPerNode: Int = 25
          val centersPartitioner: STRPartitioner = new STRPartitioner(
            centersNumPartitions,
            centersSampleRate,
            centersDimension,
            centersTransferThreshold,
            centersMaxEntriesPerNode,
            pointCenters )
          val centersMBRs = centersPartitioner.mbrBound.map{ mbr =>
              "%d;%s".format(mbr._2, mbr2wkt(mbr._1))
            }
          saveStringArrayWithoutTimeMillis(centersMBRs, "CentersMBRs", conf)
    ///////////
    
    ////////////////////////////////////////////////////////////////////
    saveStringArrayWithoutTimeMillis(p1.map(c => "%d,%.2f,%.2f".format(c.id, c.x, c.y)).collect(), "Points", conf)
    saveStringArrayWithoutTimeMillis(centers.map(c => "%d,%.2f,%.2f".format(c.id, c.x, c.y)).collect(), "Centers", conf)
    ////////////////////////////////////////////////////////////////////
  }

  def findCenters(pairs: RDD[Pair], epsilon: Double): RDD[Pair] = {
    val r2: Double = math.pow(epsilon / 2, 2)
    val centerPairs = pairs
      .map { (pair: Pair) =>
        calculateCenterCoordinates(pair, r2)
      }
    centerPairs
  }

  def calculateCenterCoordinates(pair: Pair, r2: Double): Pair = {
    var centerPair = Pair(-1, 0, 0, 0, 0, 0) //To be filtered in case of duplicates...
    val X: Double = pair.x - pair.x2
    val Y: Double = pair.y - pair.y2
    val D2: Double = math.pow(X, 2) + math.pow(Y, 2)
    if (D2 != 0.0){
      val root: Double = math.sqrt(math.abs(4.0 * (r2 / D2) - 1.0))
      val h1: Double = ((X + Y * root) / 2) + pair.x2
      val k1: Double = ((Y - X * root) / 2) + pair.y2
      val h2: Double = ((X - Y * root) / 2) + pair.x2
      val k2: Double = ((Y + X * root) / 2) + pair.y2
      centerPair = Pair(pair.id, h1, k1, pair.id2, h2, k2)
    }
    centerPair
  }

  def saveStringArrayWithoutTimeMillis(array: Array[String], tag: String, conf: Conf): Unit = {
    val path = s"$phd_home${conf.valpath()}"
    val filename = s"${conf.dataset()}_E${conf.epsilon()}"
    new java.io.PrintWriter("%s%s_%s.txt".format(path, filename, tag)) {
      write(array.mkString("\n"))
      close()
    }
  }

  def toWKT(coordinatesString: String): String = {
    val coordinates = coordinatesString.split(";")
    val min_x = coordinates(0).toDouble
    val min_y = coordinates(1).toDouble
    val max_x = coordinates(2).toDouble
    val max_y = coordinates(3).toDouble
    
    toWKT(min_x, min_y, max_x, max_y)
  }

  def toWKT(minx: Double, miny: Double, maxx: Double, maxy: Double): String = {
    "POLYGON (( %f %f, %f %f, %f %f, %f %f, %f %f ))".
      format(minx,maxy,maxx,maxy,maxx,miny,minx,miny,minx,maxy)
  }
  
  def mbr2wkt(mbr: MBR): String = toWKT(mbr.low.coord(0),mbr.low.coord(1),mbr.high.coord(0),mbr.high.coord(1))

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val epsilon:    ScallopOption[Double] = opt[Double] (default = Some(10.0))
    val partitions: ScallopOption[Int]    = opt[Int]    (default = Some(64))
    val cores:      ScallopOption[Int]    = opt[Int]    (default = Some(30))
    val master:     ScallopOption[String] = opt[String] (default = Some("spark://169.235.27.134:7077")) /* spark://169.235.27.134:7077 */
    val path:       ScallopOption[String] = opt[String] (default = Some("Y3Q1/Datasets/"))
    val valpath:    ScallopOption[String] = opt[String] (default = Some("Y3Q1/Validation/MBRs/"))
    val dataset:    ScallopOption[String] = opt[String] (default = Some("B20K"))
    val extension:  ScallopOption[String] = opt[String] (default = Some("csv"))
    verify()
  }
   
  def main(args: Array[String]): Unit = {
    // Reading arguments from command line...
    val conf = new Conf(args)
    val master = conf.master()
    // Starting session...
    var timer = System.currentTimeMillis()
    val simba = SimbaSession.builder()
      .master(master)
      .appName("PartitionViewer")
      .config("simba.index.partitions", conf.partitions().toString)
      .config("spark.cores.max", conf.cores().toString)
      .getOrCreate()
    logger.info("Starting session... [%.3fs]".format((System.currentTimeMillis() - timer)/1000.0))
    // Reading...
    timer = System.currentTimeMillis()
    phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
    val filename = "%s%s%s.%s".format(phd_home, conf.path(), conf.dataset(), conf.extension())
    val points = simba.sparkContext.
      textFile(filename, conf.cores()).
      cache()
    nPoints = points.count()
    logger.info("Reading dataset... [%.3fs]".format((System.currentTimeMillis() - timer)/1000.0))
    // Running MaximalFinder...
    logger.info("Lauching PartitionViewer...")
    val start = System.currentTimeMillis()
    PartitionViewer.run(points, simba, conf)
    val end = System.currentTimeMillis()
    logger.info("Finishing PartitionViewer...")
    logger.info("Total time for PartitionViewer: %.3fms...".format((end - start)/1000.0))
    // Closing session...
    timer = System.currentTimeMillis()
    simba.close
    logger.info("Closing session... [%.3fs]".format((System.currentTimeMillis() - timer)/1000.0))
  }
}
