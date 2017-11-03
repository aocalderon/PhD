import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}
import Misc.GeoGSON
import SPMF.{AlgoLCM, Transactions}
import JLCM.{ListCollector, TransactionsReader}
import fr.liglab.jlcm.internals.ExplorationStep;
import fr.liglab.jlcm.io.PatternsCollector;
import fr.liglab.jlcm.PLCM;
import scala.collection.JavaConverters._
import org.apache.spark.rdd.DoubleRDDFunctions
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.simba.{Dataset, SimbaSession}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.simba.index._
import org.slf4j.{Logger, LoggerFactory}
import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.joda.time.DateTime

val logger: Logger = LoggerFactory.getLogger("myLogger")
case class SP_Point(id: Int, x: Double, y: Double)
case class ACenter(id: Long, x: Double, y: Double)
case class Candidate(id: Long, x: Double, y: Double, items: String)
case class BBox(minx: Double, miny: Double, maxx: Double, maxy: Double)

  def calculateDiskCenterCoordinates(p1: SP_Point, p2: SP_Point, r2: Double): ACenter = {
    val X: Double = p1.x - p2.x
    val Y: Double = p1.y - p2.y
    var D2: Double = math.pow(X, 2) + math.pow(Y, 2)
    //var aCenter: ACenter = new ACenter(0, 0, 0)
    if (D2 != 0){
      val root: Double = math.sqrt(math.abs(4.0 * (r2 / D2) - 1.0))
      val h1: Double = ((X + Y * root) / 2) + p2.x
      val k1: Double = ((Y - X * root) / 2) + p2.y

      ACenter(0, h1, k1)
    } else {
      ACenter(0, 0, 0)
    }
  }

  def findDisks(pairsRDD: RDD[Row], epsilon: Double): RDD[ACenter] = {
    val r2: Double = math.pow(epsilon / 2, 2)
    val centers = pairsRDD.
      map { (row: Row) =>
        val p1 = SP_Point(row.getInt(0), row.getDouble(1), row.getDouble(2))
        val p2 = SP_Point(row.getInt(3), row.getDouble(4), row.getDouble(5))
        calculateDiskCenterCoordinates(p1, p2, r2)
      }
    centers
  }

    val DATASET = "B5K_Tester"
    val ENTRIES = "10"
    val PARTITIONS = "10"
    val EPSILON = 20.0
    val MU = 5
    val MASTER = "local[10]"
    val CORES = "10"
    val POINT_SCHEMA = ScalaReflection.schemaFor[SP_Point].dataType.asInstanceOf[StructType]
    val DELTA = 0.01
    val EPSG = "3068"
    // Setting session...
    logger.info("Setting session...")
    val simba = SimbaSession.builder().
      master(MASTER).
      appName("MaximalFinder").
      config("simba.rtree.maxEntriesPerNode", ENTRIES).
      config("simba.index.partitions", PARTITIONS).
      config("spark.cores.max", CORES).
      getOrCreate()
    import simba.implicits._
    import simba.simbaImplicits._
    // Reading...
    val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
    val path = "Y3Q1/Datasets/"
    val extension = "csv"
    val filename = "%s%s%s.%s".format(phd_home, path, DATASET, extension)
    logger.info("Reading %s...".format(filename))
    val points = simba.read.option("header", "false").schema(POINT_SCHEMA).csv(filename).as[SP_Point]
    val n = points.count()
    // Indexing...
    logger.info("Indexing %d points...".format(n))
    val p1 = points.toDF("id1", "x1", "y1")
    p1.index(RTreeType, "p1RT", Array("x1", "y1"))
    val p2 = points.toDF("id2", "x2", "y2")
    val epsilon = EPSILON
      val pairsRDD = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon).
        filter((row: Row) => row.getInt(0) < row.getInt(3)).
        rdd
      pairsRDD.cache()
      val npairs = pairsRDD.count()
      // Computing disks...
      timer = System.currentTimeMillis()
      val centers = findDisks(pairsRDD, epsilon).
        toDS().
        index(RTreeType, "centersRT", Array("x", "y")).
        withColumn("id", monotonically_increasing_id()).
        as[ACenter]
      centers.cache()
      val ncenters = centers.count()
      logger.info("Computing disks... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
      // Mapping disks and points...
      timer = System.currentTimeMillis()
      val candidates = centers
        .distanceJoin(p1, Array("x", "y"), Array("x1", "y1"), (epsilon / 2) + PRECISION)
        .groupBy("id", "x", "y")
        .agg(collect_list("id1").alias("IDs"))
      candidates.cache()
      val ncandidates = candidates.count()
      logger.info("Mapping disks and points... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
      // Filtering less-than-mu disks...
      timer = System.currentTimeMillis()
      val filteredCandidates = candidates.
        filter(row => row.getList(3).size() >= MU).
        map(d => (d.getLong(0), d.getDouble(1), d.getDouble(2), d.getList[Integer](3).asScala.mkString(",")))
      filteredCandidates.cache()
      /* val nFilteredCandidates = filteredCandidates.count() */
      logger.info("Filtering less-than-mu disks... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))

