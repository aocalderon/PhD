import SPMF.{AlgoFPMax, AlgoLCM, Transactions}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index.{RTree, RTreeType}
import org.apache.spark.sql.simba.partitioner.STRPartitioner
import org.apache.spark.sql.simba.spatial.{MBR, Point}
import org.joda.time.DateTime
import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

object MaximalFinderExpansion {
  private val logger: Logger = LoggerFactory.getLogger("myLogger")
  private val precision: Double = 0.001
  private val dimensions: Int = 2
  private val sampleRate: Double = 0.01
  private var phd_home: String = ""
  private var nPoints: Long = 0

  case class SP_Point(id: Long, x: Double, y: Double)
  case class P1(id1: Long, x1: Double, y1: Double)
  case class P2(id2: Long, x2: Double, y2: Double)
  case class Center(id: Long, x: Double, y: Double)
  case class Pair(id1: Long, x1: Double, y1: Double, id2: Long, x2: Double, y2: Double)
  case class Candidate(id: Long, x: Double, y: Double, items: String)
  case class Maximal(id: Long, x: Double, y: Double, items: String)
  case class CandidatePoints(cid: Long, pid: Long)
  case class MaximalPoints(partitionId: Int, maximalId: Long, pointId: Long)
  case class MaximalMBR(maximal: Maximal, mbr: MBR)
  case class BBox(minx: Double, miny: Double, maxx: Double, maxy: Double)

  def run(pointsRDD: RDD[String]
      , simba: SimbaSession
      , conf: FlockFinder.Conf): RDD[String] = {
    // 00.Setting variables...
    val mu = conf.mu()
    val epsilon = conf.epsilon()
    var maximals3: RDD[String] = simba.sparkContext.emptyRDD
    
    import simba.implicits._
    import simba.simbaImplicits._
    logger.info("00.Setting mu=%d,epsilon=%.1f,cores=%d,dataset=%s"
      .format(mu, epsilon, conf.cores(), conf.dataset()))
    val startTime = System.currentTimeMillis()
    // 01.Indexing points...
    var timer = System.currentTimeMillis()
    var pointsNumPartitions = pointsRDD.getNumPartitions
    logger.info("[Partitions Info]Points;Before indexing;%d".format(pointsNumPartitions))
    val p1 = pointsRDD.map(_.split(",")).
      map(p => P1(p(0).trim.toLong,p(1).trim.toDouble,p(2).trim.toDouble)).
      toDS().
      index(RTreeType,"p1RT",Array("x1","y1")).
      cache()
    val p2 = pointsRDD.map(_.split(",")).
      map(p => P2(p(0).trim.toLong,p(1).trim.toDouble,p(2).trim.toDouble)).
      toDS().
      index(RTreeType,"p2RT",Array("x2","y2")).
      cache()
    val points = pointsRDD.map(_.split(",")).
      map(p => SP_Point(p(0).trim.toLong,p(1).trim.toDouble,p(2).trim.toDouble)).
      toDS().
      index(RTreeType,"pointsRT",Array("x","y")).
      cache()
    pointsNumPartitions = points.rdd.getNumPartitions
    logger.info("[Partitions Info]Points;After indexing;%d".format(pointsNumPartitions))
    logger.info("01.Indexing points... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nPoints))
    // 02.Getting pairs...
    timer = System.currentTimeMillis()
    val pairs = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon + precision)
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
      var candidatesNumPartitions: Int = candidates.rdd.getNumPartitions
      logger.info("[Partitions Info]Candidates;Before indexing;%d".format(candidatesNumPartitions))
      val pointCandidate = candidates.map{ candidate =>
          ( new Point(Array(candidate.x, candidate.y)), candidate)
        }
        .rdd
        .cache()
      val candidatesSampleRate: Double = sampleRate
      val candidatesDimension: Int = dimensions
      val candidatesTransferThreshold: Long = 800 * 1024 * 1024
      val candidatesMaxEntriesPerNode: Int = 25
      candidatesNumPartitions = conf.cores()
      val candidatesPartitioner: STRPartitioner = new STRPartitioner(candidatesNumPartitions
        , candidatesSampleRate
        , candidatesDimension
        , candidatesTransferThreshold
        , candidatesMaxEntriesPerNode
        , pointCandidate)
      logger.info("08.Indexing candidates... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidates))
      // 09.Getting expansions...
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
      val nCandidates2 = candidates2.count()
      candidatesNumPartitions = candidates2.getNumPartitions
      logger.info("[Partitions Info]Candidates;After indexing;%d".format(candidatesNumPartitions))
      logger.info("09.Getting expansions... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidates2))
      // 10.Finding maximal disks...
      timer = System.currentTimeMillis()
      val method = conf.method()
      val maximals = candidates2
        .mapPartitionsWithIndex{ (partitionIndex, partitionCandidates) =>
          var maximalsIterator: Iterator[(Int, List[Long])] = null
          if(method == "fpmax"){
            val transactions = partitionCandidates
              .map { candidate =>
                candidate.items
                .split(" ")
                .map(new Integer(_))
                .toList.asJava
              }.toList.asJava
            val algorithm = new AlgoFPMax
            val maximals = algorithm.runAlgorithm(transactions, 1)
            maximalsIterator = maximals.getItemsets(mu)
              .asScala
              .map(m => (partitionIndex, m.asScala.toList.map(_.toLong).sorted))
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
              .map(m => (partitionIndex, m.asScala.toList.map(_.toLong).sorted))
              .toIterator
          }
          maximalsIterator
        }
        .cache()
      var nMaximals = maximals.map(_._2.mkString(" ")).distinct().count()
      logger.info("10.Finding maximal disks... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nMaximals))
      // 11.Prunning duplicates and subsets...
      timer = System.currentTimeMillis()
      val EMBRs = expandedMBRs.map{ mbr =>
          mbr._2 -> "%f;%f;%f;%f".format(mbr._1.low.coord(0),mbr._1.low.coord(1),mbr._1.high.coord(0),mbr._1.high.coord(1))
        }
        .toMap
      val maximalPoints = maximals
        .toDF("partitionId", "pointsId")
        .withColumn("maximalId", monotonically_increasing_id())
        .withColumn("pointId", explode($"pointsId"))
        .select("partitionId", "maximalId", "pointId")
        .as[MaximalPoints]
        .cache()
      val maximals2 = maximalPoints
        .join(points, maximalPoints.col("pointId") === points.col("id"))
        .groupBy($"partitionId", $"maximalId").agg(min($"x"), min($"y"), max($"x"), max($"y"), collect_list("pointId"))
        .map{ m => 
          val MBRCoordinates = EMBRs(m.getInt(0))
          val x = (m.getDouble(2) + m.getDouble(4)) / 2.0
          val y = (m.getDouble(3) + m.getDouble(5)) / 2.0
          val pids = m.getList[Long](6).asScala.sorted.mkString(" ")
          val maximal = "%f;%f;%s".format(x, y, pids)
          (maximal, MBRCoordinates)
        }
        .map(m => (m._1, m._2, isNotInExpansionArea(m._1, m._2, epsilon)))
        .cache()
      maximals3 = maximals2
        .filter(m => m._3)
        .map(m => m._1)   
        .distinct()
        .rdd
      nMaximals = maximals3.count()
      logger.info("11.Prunning duplicates and subsets... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nMaximals))
      val endTime = System.currentTimeMillis()
      val totalTime = (endTime - startTime)/1000.0
      // Printing info summary ...
      logger.info("%12s,%6s,%6s,%3s,%7s,%8s,%10s,%13s,%11s".
        format("Dataset", "Eps", "Cores", "Mu", "Time",
          "# Pairs", "# Disks", "# Candidates", "# Maximals"
        )
      )
      logger.info("%12s,%6.1f,%6d,%3d,%7.2f,%8d,%10d,%13d,%11d".
        format( conf.dataset(), conf.epsilon(), conf.cores(), conf.mu(), totalTime,
          nPairs, nDisks, nCandidates, nMaximals
        )
      )
    }
    // Dropping indices...
    timer = System.currentTimeMillis()
    p1.dropIndex()
    p2.dropIndex()
    points.dropIndex()
    centers.dropIndex()
    logger.info("Dropping indices...[%.3fs]".format((System.currentTimeMillis() - timer)/1000.0))
    
    maximals3
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

  def isNotInExpansionArea(maximalString: String, coordinatesString: String, epsilon: Double): Boolean = {
    val maximal = maximalString.split(";")
    val x = maximal(0).toDouble
    val y = maximal(1).toDouble
    val coordinates = coordinatesString.split(";")
    val min_x = coordinates(0).toDouble
    val min_y = coordinates(1).toDouble
    val max_x = coordinates(2).toDouble
    val max_y = coordinates(3).toDouble

    x <= (max_x - epsilon) &&
      x >= (min_x + epsilon) &&
      y <= (max_y - epsilon) &&
      y >= (min_y + epsilon)
  }

  def isInside(x: Double, y: Double, bbox: BBox, epsilon: Double): Boolean = {
    x < (bbox.maxx - epsilon) &&
      x > (bbox.minx + epsilon) &&
      y < (bbox.maxy - epsilon) &&
      y > (bbox.miny + epsilon)
  }

  def saveStringArray(array: Array[String], tag: String, conf: Conf): Unit = {
    val path = s"$phd_home${conf.valpath()}"
    val filename = s"${conf.dataset()}_E${conf.epsilon()}_M${conf.mu()}_C${conf.cores()}"
    new java.io.PrintWriter("%s%s_%s_%d.txt".format(path, filename, tag, System.currentTimeMillis)) {
      write(array.mkString("\n"))
      close()
    }
  }

  def saveStringArrayWithoutTimeMillis(array: Array[String], tag: String, conf: Conf): Unit = {
    val path = s"$phd_home${conf.valpath()}"
    val filename = s"${conf.dataset()}_E${conf.epsilon()}_M${conf.mu()}_C${conf.cores()}"
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
    val mu:         ScallopOption[Int]    = opt[Int]    (default = Some(5))
    val entries:    ScallopOption[Int]    = opt[Int]    (default = Some(25))
    val partitions: ScallopOption[Int]    = opt[Int]    (default = Some(1024))
    val candidates: ScallopOption[Int]    = opt[Int]    (default = Some(256))
    val cores:      ScallopOption[Int]    = opt[Int]    (default = Some(28))
    val master:     ScallopOption[String] = opt[String] (default = Some("spark://169.235.27.134:7077")) /* spark://169.235.27.134:7077 */
    val path:       ScallopOption[String] = opt[String] (default = Some("Y3Q1/Datasets/"))
    val valpath:    ScallopOption[String] = opt[String] (default = Some("Y3Q1/Validation/"))
    val dataset:    ScallopOption[String] = opt[String] (default = Some("B20K"))
    val extension:  ScallopOption[String] = opt[String] (default = Some("csv"))
    val method:     ScallopOption[String] = opt[String] (default = Some("fpmax"))
    // FlockFinder parameters
    val delta:	    ScallopOption[Int]    = opt[Int]    (default = Some(3))    
    val tstart:     ScallopOption[Int]    = opt[Int]    (default = Some(0))
    val tend:       ScallopOption[Int]    = opt[Int]    (default = Some(5))
    val cartesian:  ScallopOption[Int]    = opt[Int]    (default = Some(2))
    val logs:	    ScallopOption[String] = opt[String] (default = Some("INFO"))    
    
    verify()
  }

  def main(args: Array[String]): Unit = {
    // Reading arguments from command line...
    val conf = new FlockFinder.Conf(args)
    val master = conf.master()
    // Starting session...
    var timer = System.currentTimeMillis()
    val simba = SimbaSession.builder()
      .master(master)
      .appName("MaximalFinderExpansion")
      .config("simba.index.partitions",conf.partitions().toString)
      .config("spark.cores.max",conf.cores().toString)
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
