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

  case class SP_Point(id: Int, x: Double, y: Double)
  case class ACenter(id: Long, x: Double, y: Double)
  case class Candidate(id: Long, x: Double, y: Double, items: String)
  case class BBox(minx: Double, miny: Double, maxx: Double, maxy: Double)

  def run(points: Dataset[MaximalFinder2.SP_Point],
          simba: SimbaSession,
          conf: MaximalFinder2.Conf) = {
    import simba.implicits._
    import simba.simbaImplicits._
    logger.info("Setting mu=%d,epsilon=%.1f,cores=%d,partitions=%d,dataset=%s".
      format(conf.mu(),conf.epsilon(),conf.cores(),conf.partitions(),conf.dataset()))
    // Indexing...
    var timer = System.currentTimeMillis()
    val p1 = points.toDF("id1", "x1", "y1")
    p1.index(RTreeType, "p1RT", Array("x1", "y1"))
    val p2 = points.toDF("id2", "x2", "y2")
    logger.info("Indexing... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
    // Self-join...
    timer = System.currentTimeMillis()
    val pairs = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), conf.epsilon()).
      rdd.
      filter((pair: Row) => pair.getInt(0) < pair.getInt(3))
    pairs.cache()
    val nPairs = pairs.count()
    pairs.take(10).foreach(println)
    logger.info("Getting pairs... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))

    // Dropping point indices...
    timer = System.currentTimeMillis()
    p1.dropIndexByName("p1RT")
    logger.info("Dropping point indices...[%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
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
    points.count()
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