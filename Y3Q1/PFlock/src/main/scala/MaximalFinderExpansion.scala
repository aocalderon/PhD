import SPMF.{AlgoFPMax, AlgoLCM, Transactions}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.functions._
import org.apache.spark.sql.simba.index.{RTree, RTreeType}
import org.apache.spark.sql.simba.partitioner.STRPartitioner
import org.apache.spark.sql.simba.spatial.{MBR, Point, Shape}
import org.apache.spark.sql.simba.{Dataset, SimbaSession}
import org.apache.spark.sql.types.StructType
import org.joda.time.DateTime
import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

object MaximalFinderExpansion {
  private val logger: Logger = LoggerFactory.getLogger("myLogger")
  private val precision: Double = 0.001
  private val dimensions: Int = 2
  private val sampleRate: Double = 0.01
  private var nPoints: Long = 0

  case class SP_Point(id: Long, x: Double, y: Double)
  case class Center(id: Long, x: Double, y: Double)
  case class Pair(id1: Long, x1: Double, y1: Double, id2: Long, x2: Double, y2: Double)
  case class Candidate(id: Long, x: Double, y: Double, items: String)
  case class Maximal(x: Double, y: Double, items: String)
  case class CandidatePoints(cid: Long, pid: Long)
  case class MaximalPoints(mid: Long, pid: Long)
  case class BBox(minx: Double, miny: Double, maxx: Double, maxy: Double)

  def run(points: Dataset[SP_Point]
      , simba: SimbaSession
      , conf: Conf): Unit = {
    // 00.Setting variables...
    val mu = conf.mu()
    val epsilon = conf.epsilon()
    import simba.implicits._
    import simba.simbaImplicits._
    logger.info("00.Setting mu=%d,epsilon=%.1f,cores=%d,dataset=%s"
      .format(mu, epsilon, conf.cores(), conf.dataset()))
    val startTime = System.currentTimeMillis()
    // 01.Indexing points...
    var timer = System.currentTimeMillis()
    val p1 = points.toDF("id1", "x1", "y1")
    p1.index(RTreeType, "p1RT", Array("x1", "y1")).cache()
    val p2 = points.toDF("id2", "x2", "y2")
    p2.index(RTreeType, "p2RT", Array("x2", "y2")).cache()
    logger.info("01.Indexing points... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nPoints))
    // 02.Getting pairs...
    timer = System.currentTimeMillis()
    val pairs = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon)
      .as[Pair]
      .filter(pair => pair.id1 < pair.id2)
      .rdd
      .cache()
    val nPairs = pairs.count()
    logger.info("02.Getting pairs... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nPairs))
    // 03.Computing centers...
    timer = System.currentTimeMillis()
    val centerPairs = findCenters(pairs, epsilon)
      .filter( pair => pair.id1 != -1 )
      .toDS()
      .as[Pair]
      .cache()
    val leftCenters = centerPairs.select("x1","y1")
    val rightCenters = centerPairs.select("x2","y2")
    val centers = leftCenters.union(rightCenters)
      .toDF("x", "y")
      .withColumn("id", monotonically_increasing_id())
      .as[SP_Point]
      .cache()
    val nCenters = centers.count()
    logger.info("03.Computing centers... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCenters))
    // 04.Indexing centers...
    timer = System.currentTimeMillis()
    centers.index(RTreeType, "centersRT", Array("x", "y")).cache()
    logger.info("04.Indexing centers... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCenters))
    // 05.Getting disks...
    timer = System.currentTimeMillis()
    val disks = centers
      .distanceJoin(p1, Array("x", "y"), Array("x1", "y1"), (epsilon / 2) + precision)
      .groupBy("id", "x", "y")
      .agg(collect_list("id1").alias("ids"))
      .cache()
    val nDisks = disks.count()
    logger.info("05.Getting disks... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nDisks))
    // 06.Filtering less-than-mu disks...
    timer = System.currentTimeMillis()
    val filteredDisks = disks
      .filter(row => row.getList[Long](3).size() >= mu)
      .rdd
      .cache()
    val nFilteredDisks = filteredDisks.count()
    logger.info("06.Filtering less-than-mu disks... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nFilteredDisks))
    // 07.Prunning duplicate candidates...
    timer = System.currentTimeMillis()
    val candidatePoints = filteredDisks
      .map{ c =>
        ( c.getLong(0), c.getList[Long](3).asScala)
      }
      .toDF("cid", "items")
      .withColumn("pid", explode($"items"))
      .select("cid","pid")
      .as[CandidatePoints]
      .cache()
    val candidates = candidatePoints
      .join(points, candidatePoints.col("pid") === points.col("id"))
      .groupBy($"cid").agg(min($"x"), min($"y"), max($"x"), max($"y"), collect_list("pid"))
      .map{ c => 
        Candidate( 0
          , (c.getDouble(1) + c.getDouble(3)) / 2.0
          , (c.getDouble(2) + c.getDouble(4)) / 2.0
          , c.getList[Long](5).asScala.sorted.mkString(" ") 
        )
      }
      .distinct()
      .cache()
    val nCandidates = candidates.count()
    logger.info("07.Prunning duplicate candidates... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidates))
    // 08.Indexing candidates... 
    if(nCandidates > 0){
      val pointCandidate = candidates.map{ candidate =>
          ( new Point(Array(candidate.x, candidate.y)), candidate)
        }
        .rdd
        .cache()
      val candidatesSampleRate: Double = sampleRate
      val candidatesDimension: Int = dimensions
      val candidatesTransferThreshold: Long = 800 * 1024 * 1024
      val candidatesMaxEntriesPerNode: Int = 25
      //val candidatesPartitionSize: Int = conf.candidates()
      //var candidatesNumPartitions: Int = Math.ceil(nCandidates / candidatesPartitionSize).toInt
      var candidatesNumPartitions: Int = conf.cores()
      logger.info("Candidates # of partitions: %d".format(candidates.rdd.getNumPartitions))
      val candidatesPartitioner: STRPartitioner = new STRPartitioner(candidatesNumPartitions
        , candidatesSampleRate
        , candidatesDimension
        , candidatesTransferThreshold
        , candidatesMaxEntriesPerNode
        , pointCandidate)
      logger.info("08.Indexing candidates... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidates))
      // 09.Getting expansions...

      ////////////////////////////////////////////////////////////////
      saveStringArray(candidatesPartitioner.mbrBound.map(mbr => "%d;%s".format(mbr._2, mbr2wkt(mbr._1))), "MBRs", conf)
      ////////////////////////////////////////////////////////////////

      timer = System.currentTimeMillis()
      val expandedMBRs = candidatesPartitioner.mbrBound
        .map{ mbr =>
          val mins = new Point( Array(mbr._1.low.coord(0) - epsilon
            , mbr._1.low.coord(1) - epsilon) )
          val maxs = new Point( Array(mbr._1.high.coord(0) + epsilon
            , mbr._1.high.coord(1) + epsilon) )
          ( MBR(mins, maxs), mbr._2, 1 )
        }
      val expandedRTree = RTree(expandedMBRs, candidatesMaxEntriesPerNode)
      candidatesNumPartitions = expandedMBRs.length
      val candidates2 = pointCandidate.flatMap{ candidate =>
        expandedRTree.circleRange(candidate._1, 0.0)
          .map{ mbr => 
            ( mbr._2, candidate._2 )
          }
        }
        .partitionBy(new ExpansionPartitioner(candidatesNumPartitions))
        .map(_._2)
        .cache()
      candidatesNumPartitions = candidates2.getNumPartitions
      logger.info("Candidates # of partitions: %d".format(candidatesNumPartitions))
      val nCandidates2 = candidates2.count()
      logger.info("09.Getting expansions... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidates2))
      // 10.Finding maximal disks...

      ////////////////////////////////////////////////////////////////
      saveStringArray(expandedMBRs.map(mbr => "%d;%s".format(mbr._2, mbr2wkt(mbr._1))), "EMBRs", conf)
      ////////////////////////////////////////////////////////////////

      timer = System.currentTimeMillis()
      val method = conf.method()
      val maximals = candidates2
        .mapPartitionsWithIndex{ (_, partitionCandidates) =>
          var maximalsIterator: Iterator[List[Long]] = null
          if(method == "fpmax"){
            val transactions = partitionCandidates
              .map { candidate =>
                candidate.items
                .split(" ")
                .map(new Integer(_))
                .toList.asJava
              }.toList.asJava
            ////////////////////////////////////////////////////////
            //partition.map(t => "%d;%f;%f;%s;%s;%d".format(index, t._1.x, t._1.y, t._1.items, t._2.toString, t._3))
            ////////////////////////////////////////////////////////
            val algorithm = new AlgoFPMax
            val maximals = algorithm.runAlgorithm(transactions, 1)
            maximalsIterator = maximals.getItemsets(mu)
              .asScala
              .map(m => m.asScala.toList.map(_.toLong))
              .toIterator
          } else {
            val transactions = partitionCandidates
              .map { candidate =>
                candidate.items
                .split(" ")
                .map(new Integer(_))
                .toList.asJava
              }.toSet.asJava
            val LCM = new AlgoLCM
            val data = new Transactions(transactions)
            val closed = LCM.runAlgorithm(1, data)
            maximalsIterator = closed.getMaximalItemsets1(mu)
              .asScala
              .map(m => m.asScala.toList.map(_.toLong))
              .toIterator
          }
          maximalsIterator
        }
        .cache()
      var nMaximals = maximals.count()
      logger.info("10.Finding maximal disks... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nMaximals))
      // 11.Prunning duplicates...
      timer = System.currentTimeMillis()
      val maximalPoints = maximals
        .toDF("pids")
        .withColumn("mid", monotonically_increasing_id())
        .withColumn("pid", explode($"pids"))
        .select("mid", "pid")
        .as[MaximalPoints]
        .cache()
      val centerMaximal = maximalPoints
        .join(points, maximalPoints.col("pid") === points.col("id"))
        .groupBy($"mid").agg(min($"x"), min($"y"), max($"x"), max($"y"), collect_list("pid"))
        .map{ m => 
          val x = (m.getDouble(1) + m.getDouble(3)) / 2.0
          val y = (m.getDouble(2) + m.getDouble(4)) / 2.0
          val items = m.getList[Long](5).asScala.sorted.mkString(" ")
          val center = new Point(Array(x, y)) 
          val maximal = Maximal(x, y, items)
          (center, maximal)
        }
        .cache()
       val maximals2 = centerMaximal.flatMap{ maximal =>
         expandedRTree.circleRange(maximal._1, 0.0)
          .map{ mbr => 
            ( maximal._2, isInExpansionArea(maximal._2, mbr._1, epsilon) ) 
          }
        }
        .filter(maximal => !maximal._2)
        .map(maximal => maximal._1)   
        .distinct()
        .cache()
      nMaximals = maximals2.count()
      logger.info("11.Prunning duplicates... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nMaximals))
      val endTime = System.currentTimeMillis()
      val totalTime = (endTime - startTime)/1000.0

      ////////////////////////////////////////////////////////////////
      saveStringArray(maximals2.map(m => "%f;%f;%s".format(m.x, m.y, m.items)).collect(), "Maximals", conf)
      ////////////////////////////////////////////////////////////////
      
      // Printing info summary ...
      logger.info("%12s,%6s,%6s,%7s,%8s,%10s,%13s,%11s".
        format("Dataset", "Eps", "Cores", "Time",
          "# Pairs", "# Disks", "# Candidates", "# Maximals"
        )
      )
      logger.info("%12s,%6.1f,%6d,%7.2f,%8d,%10d,%13d,%11d".
        format( conf.dataset(), conf.epsilon(), conf.cores(), totalTime,
          nPairs, nDisks, nCandidates, nMaximals
        )
      )
    }
    // Dropping indices...
    timer = System.currentTimeMillis()
    p1.dropIndex()
    p2.dropIndex()
    centers.dropIndex()
    logger.info("Dropping indices...[%.3fs]".format((System.currentTimeMillis() - timer)/1000.0))
  }

  import org.apache.spark.Partitioner
  class ExpansionPartitioner(partitions: Long) extends Partitioner{
    override def numPartitions: Int = partitions.toInt

    override def getPartition(key: Any): Int = {
      key.asInstanceOf[Int]
    }
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
    val X: Double = pair.x1 - pair.x2
    val Y: Double = pair.y1 - pair.y2
    val D2: Double = math.pow(X, 2) + math.pow(Y, 2)
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

  def isInExpansionArea(maximal: Maximal, shape: Shape, epsilon: Double): Boolean = {
    val mbr = shape.asInstanceOf[MBR]
    val x = maximal.x
    val y = maximal.y
    x < (mbr.high.coord(0) - epsilon) &&
      x > (mbr.low.coord(0) + epsilon) &&
      y < (mbr.high.coord(1) - epsilon) &&
      y > (mbr.low.coord(1) + epsilon)
    
  }

  def isInside(x: Double, y: Double, bbox: BBox, epsilon: Double): Boolean = {
    x < (bbox.maxx - epsilon) &&
      x > (bbox.minx + epsilon) &&
      y < (bbox.maxy - epsilon) &&
      y > (bbox.miny + epsilon)
  }

  def saveStringArray(array: Array[String], filename: String, conf: Conf): Unit = {
    new java.io.PrintWriter("/tmp/%s_E%.1f_M%d_%s.txt".format(conf.dataset(), conf.epsilon(), conf.mu(), filename)) {
      write(array.mkString("\n"))
      close()
    }
  }

  def toWKT(minx: Double, miny: Double, maxx: Double, maxy: Double): String = {
    "POLYGON (( %f %f, %f %f, %f %f, %f %f, %f %f ))".
      format(minx, maxy,maxx, maxy,maxx, miny,minx, miny,minx, maxy)
  }
  
  def mbr2wkt(mbr: MBR): String = toWKT(mbr.low.coord(0),mbr.low.coord(1),mbr.high.coord(0),mbr.high.coord(1))

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val epsilon:  ScallopOption[Double]   = opt[Double](default = Some(10.0))
    val mu:     ScallopOption[Int]    = opt[Int]   (default = Some(5))
    val entries:	ScallopOption[Int]    = opt[Int]   (default = Some(25))
    val partitions:	ScallopOption[Int]    = opt[Int]   (default = Some(1024))
    val candidates:	ScallopOption[Int]    = opt[Int]   (default = Some(256))
    val cores:    ScallopOption[Int]    = opt[Int]   (default = Some(28))
    val master:   ScallopOption[String]	= opt[String](default = Some("spark://169.235.27.138:7077")) /* spark://169.235.27.138:7077 */
    val path:     ScallopOption[String]	= opt[String](default = Some("Y3Q1/Datasets/"))
    val dataset:	ScallopOption[String]	= opt[String](default = Some("B20K"))
    val extension:	ScallopOption[String]	= opt[String](default = Some("csv"))
    val method:   ScallopOption[String]	= opt[String](default = Some("fpmax"))
    verify()
  }

  def main(args: Array[String]): Unit = {
    // Reading arguments from command line...
    val conf = new Conf(args)
    val POINT_SCHEMA = ScalaReflection.schemaFor[SP_Point].dataType.asInstanceOf[StructType]
    val master = conf.master()
    // Starting session...
    var timer = System.currentTimeMillis()
    val simba = SimbaSession.builder()
      .master(master)
      .appName("MaximalFinderExpansion")
      .config("simba.index.partitions",conf.partitions().toString)
      .config("spark.cores.max",conf.cores().toString)
      .getOrCreate()
    import simba.implicits._
    import simba.simbaImplicits._
    logger.info("Starting session... [%.3fs]".format((System.currentTimeMillis() - timer)/1000.0))
    // Reading...
    timer = System.currentTimeMillis()
    val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
    val filename = "%s%s%s.%s".format(phd_home, conf.path(), conf.dataset(), conf.extension())
    val points = simba.read
      .option("header", "false")
      .schema(POINT_SCHEMA)
      .csv(filename)
      .as[SP_Point]
      .cache()
    nPoints = points.count()
    logger.info("Reading dataset... [%.3fs]".format((System.currentTimeMillis() - timer)/1000.0))
    // Running MaximalFinder...
    logger.info("Lauching MaximalFinder at %s...".format(DateTime.now.toLocalTime.toString))
    val start = System.currentTimeMillis()
    MaximalFinderExpansion.run(points, simba, conf)
    val end = System.currentTimeMillis()
    logger.info("Finishing MaximalFinder at %s...".format(DateTime.now.toLocalTime.toString))
    logger.info("Total time for MaximalFinder: %.3fms...".format((end - start)/1000.0))
    // Closing session...
    timer = System.currentTimeMillis()
    simba.close
    logger.info("Closing session... [%.3fs]".format((System.currentTimeMillis() - timer)/1000.0))
  }
}
