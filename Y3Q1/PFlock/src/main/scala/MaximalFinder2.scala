import scala.collection.JavaConverters._
import org.apache.spark.rdd.DoubleRDDFunctions
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.simba.{Dataset, SimbaSession}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.simba.index._
import org.joda.time.DateTime
import org.slf4j.{Logger, LoggerFactory}
import org.rogach.scallop.{ScallopConf, ScallopOption}

object MaximalFinder2 {
  private val logger: Logger = LoggerFactory.getLogger("myLogger")
  private val precision: Double = 0.001
  private var n: Long = 0

  case class SP_Point(id: Long, x: Double, y: Double)
  case class Center(id: Long, x: Double, y: Double)
  case class Pair(id1: Long, x1: Double, y1: Double, id2: Long, x2: Double, y2: Double)
  case class Candidate(id: Long, x: Double, y: Double, items: String)
  case class BBox(minx: Double, miny: Double, maxx: Double, maxy: Double)

  def run(points: Dataset[MaximalFinder2.SP_Point],
          simba: SimbaSession,
          conf: MaximalFinder2.Conf) = {
    import simba.implicits._
    import simba.simbaImplicits._
    logger.info("0.Setting mu=%d,epsilon=%.1f,cores=%d,partitions=%d,dataset=%s".
      format(conf.mu(),conf.epsilon(),conf.cores(),conf.partitions(),conf.dataset()))
    val startTime = System.currentTimeMillis()
    // 1.Indexing points...
    var timer = System.currentTimeMillis()
    val p1 = points.toDF("id1", "x1", "y1")
    p1.index(RTreeType, "p1RT", Array("x1", "y1"))
    val p2 = points.toDF("id2", "x2", "y2")
    logger.info("1.Indexing points... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, n))
    // 2.Getting pairs...
    timer = System.currentTimeMillis()
    val pairs = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), conf.epsilon())
        .as[Pair]
        .filter(pair => pair.id1 < pair.id2)
        .rdd
    pairs.cache()
    val nPairs = pairs.count()
	logger.info("2.Getting pairs... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nPairs))
	// 3.Computing centers...
    timer = System.currentTimeMillis()
	val centerPairs = findDisks(pairs, conf.epsilon())
        .filter( pair => pair.id1 != -1 )
        .toDS()
        .as[Pair]
    centerPairs.cache()
    val leftCenters = centerPairs.select("id1","x1","y1")
    val rightCenters = centerPairs.select("id2","x2","y2")
    val centers = leftCenters.union(rightCenters)
        .toDF("id","x","y")
        .as[SP_Point]
    centers.cache()
    val nCenters = centers.count()
	logger.info("3.Computing centers... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCenters))
	// 4.Indexing centers...
    timer = System.currentTimeMillis()
    centers.index(RTreeType, "centersRT", Array("x", "y"))
	logger.info("4.Indexing centers... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCenters))
	// 5.Getting disks...
    timer = System.currentTimeMillis()
	val disks = centers
        .distanceJoin(p1, Array("x", "y"), Array("x1", "y1"), (conf.epsilon() / 2) + precision)
        .groupBy("id", "x", "y")
        .agg(collect_list("id1").alias("ids"))
	disks.cache()
    val nDisks = disks.count()
	logger.info("5.Getting disks... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nDisks))
    // 6.Filtering less-than-mu disks...
    timer = System.currentTimeMillis()
    val mu = conf.mu()
    var candidates = disks
        .filter( row => row.getList[Int](3).size() >= mu )
        .map(d => Candidate(d.getLong(0), d.getDouble(1), d.getDouble(2), d.getList[Int](3).asScala.mkString(",")))
    candidates.cache()
    val nCandidates = candidates.count()
    logger.info("6.Filtering less-than-mu disks... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidates))
    // 7.Indexing candidates...
    timer = System.currentTimeMillis()
    candidates.index(RTreeType, "filteredCandidatesRT", Array("x", "y"))
    logger.info("7.Indexing candidates... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidates))    
    
    val endTime = System.currentTimeMillis()
    val totalTime = (endTime - startTime)/1000.0
    // Printing info summary ...
    logger.info("%12s,%6s,%6s,%7s,%8s,%10s,%10s,%13s".
        format("Dataset", "Eps", "Cores", "Time",
            "# Pairs", "# Centers", "# Disks", "# Candidates"
        )
    )
    logger.info("%12s,%6.1f,%6d,%7.2f,%8d,%10d,%10d,%13d".
        format( conf.dataset(), conf.epsilon(), conf.cores(), totalTime,
            nPairs, nCenters, nDisks, nCandidates
        )
    )
    // Dropping point indices...
    timer = System.currentTimeMillis()
    p1.dropIndexByName("p1RT")
    centers.dropIndexByName("centersRT")
    logger.info("Dropping point indices...[%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
  }

  def findDisks(pairs: RDD[Pair], epsilon: Double): RDD[Pair] = {
    val r2: Double = math.pow(epsilon / 2, 2)
    val centerPairs = pairs
      .map { (pair: Pair) =>
        calculateDiskCenterCoordinates(pair, r2)
      }
      
    centerPairs
  }

  def calculateDiskCenterCoordinates(pair: Pair, r2: Double): Pair = {
    var centerPair = Pair(-1, 0, 0, 0, 0, 0) //To be filtered in case of duplicates...
    val X: Double = pair.x1 - pair.x2
    val Y: Double = pair.y1 - pair.y2
    var D2: Double = math.pow(X, 2) + math.pow(Y, 2)
    if (D2 != 0.0){
		val root: Double = math.sqrt(math.abs(4.0 * (r2 / D2) - 1.0))
		val h1: Double = ((X + Y * root) / 2) + pair.x2
		val k1: Double = ((Y - X * root) / 2) + pair.y2
		val h2: Double = ((X - Y * root) / 2) + pair.x2
		val k2: Double = ((Y + X * root) / 2) + pair.y2
	
		centerPair = Pair(pair.id1, h1, k1, pair.id2, h2, k2)
	}
    
    centerPair
  }

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val epsilon:	ScallopOption[Double]	= opt[Double](default = Some(10.0))
    val mu:		ScallopOption[Int]	= opt[Int]   (default = Some(5))
    val entries:	ScallopOption[Int]	= opt[Int]   (default = Some(25))
    val partitions:	ScallopOption[Int]	= opt[Int]   (default = Some(1024))
    val cores:	ScallopOption[Int]	= opt[Int]   (default = Some(28))
    val master:	ScallopOption[String]	= opt[String](default = Some("spark://169.235.27.138:7077"))
    val path:	ScallopOption[String]	= opt[String](default = Some("Y3Q1/Datasets/"))
    val dataset:	ScallopOption[String]	= opt[String](default = Some("B20K"))
    val extension:	ScallopOption[String]	= opt[String](default = Some("csv"))
    verify()
  }

  def main(args: Array[String]): Unit = {
    // Reading arguments from command line...
    val conf = new Conf(args)
    val POINT_SCHEMA = ScalaReflection.schemaFor[SP_Point].dataType.asInstanceOf[StructType]
    val MASTER = conf.master()
    // Starting session...
    var timer = System.currentTimeMillis()
    val simba = SimbaSession.builder().
      master(MASTER).
      appName("MaximalFinder2").
      config("simba.rtree.maxEntriesPerNode", conf.entries().toString).
      config("simba.index.partitions", conf.partitions().toString).
      config("spark.cores.max", conf.cores().toString).
      getOrCreate()
    import simba.implicits._
    import simba.simbaImplicits._
    logger.info("Starting session... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
    // Reading...
    timer = System.currentTimeMillis()
    val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
    val filename = "%s%s%s.%s".format(phd_home, conf.path(), conf.dataset(), conf.extension())
    val points = simba.read.option("header", "false").schema(POINT_SCHEMA).csv(filename).as[SP_Point]
    n = points.count()
    points.cache()
    logger.info("Reading dataset... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
    // Running MaximalFinder...
    logger.info("Lauching MaximalFinder at %s...".format(DateTime.now.toLocalTime.toString))
    val start = System.currentTimeMillis()
    MaximalFinder2.run(points, simba, conf)
    val end = System.currentTimeMillis()
    logger.info("Finishing MaximalFinder at %s...".format(DateTime.now.toLocalTime.toString))
    logger.info("Total time for MaximalFinder: %.3fms...".format((end - start)/1000.0))
    // Closing session...
    timer = System.currentTimeMillis()
    simba.close
    logger.info("Closing session... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
  }
}
