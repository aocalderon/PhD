import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.functions._
import org.apache.spark.sql.simba.index.{RTree, RTreeType}
import org.apache.spark.sql.simba.partitioner.STRPartitioner
import org.apache.spark.sql.simba.spatial.{MBR, Point}
import org.apache.spark.sql.simba.{Dataset, SimbaSession}
import org.apache.spark.sql.types.StructType
import org.joda.time.DateTime
import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

object MaximalFinderExpansionDebugger {
  private val logger: Logger = LoggerFactory.getLogger("myLogger")
  private val precision: Double = 0.001
  private var n: Long = 0

  case class SP_Point(id: Long, x: Double, y: Double)
  case class Center(id: Long, x: Double, y: Double)
  case class Pair(id1: Long, x1: Double, y1: Double, id2: Long, x2: Double, y2: Double)
  case class Candidate(id: Long, x: Double, y: Double, items: String)
  case class BBox(minx: Double, miny: Double, maxx: Double, maxy: Double)

  def run(points: Dataset[MaximalFinderExpansionDebugger.SP_Point]
          , simba: SimbaSession
          , conf: MaximalFinderExpansionDebugger.Conf): Unit = {
    val mu = conf.mu()
    val epsilon = conf.epsilon()
    import simba.implicits._
    import simba.simbaImplicits._
    logger.info("00.Setting mu=%d,epsilon=%.1f,cores=%d,partitions=%d,dataset=%s".
      format(mu, epsilon, conf.cores(), conf.partitions(), conf.dataset()))
    //val startTime = System.currentTimeMillis()
    // 1.Indexing points...
    var timer = System.currentTimeMillis()
    val p1 = points.toDF("id1", "x1", "y1")
    p1.index(RTreeType, "p1RT", Array("x1", "y1"))
    val p2 = p1.toDF("id2", "x2", "y2")
    logger.info("01.Indexing points... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, n))
    // 02.Getting pairs...
    timer = System.currentTimeMillis()
    val pairs = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon)
      .as[Pair]
      .filter(pair => pair.id1 < pair.id2)
      .rdd
    pairs.cache()
    val nPairs = pairs.count()
    logger.info("02.Getting pairs... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nPairs))
    // 03.Computing centers...
    timer = System.currentTimeMillis()
    val centerPairs = findCenters(pairs, epsilon)
      .filter( pair => pair.id1 != -1 )
      .toDS()
      .as[Pair]
    centerPairs.cache()
    val leftCenters = centerPairs.select("x1","y1")
    val rightCenters = centerPairs.select("x2","y2")
    var centers = leftCenters.union(rightCenters)
      .toDF("x","y")
      .withColumn("id", monotonically_increasing_id())
      .as[SP_Point]
      .map(c => (new Point(Array(c.x, c.y)), c))
      .rdd
    centers.cache()
    val nCenters = centers.count()
    logger.info("03.Computing centers... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCenters))
    // 04.Indexing centers...
    timer = System.currentTimeMillis()
    val centersPartitionSize: Int = conf.centers()
    val centersNumPartitions: Int = Math.ceil(nCenters / centersPartitionSize).toInt
    val centersSampleRate: Double = 0.05
    val centersDimension: Int = 2
    val centersTransferThreshold: Long = 800 * 1024 * 1024
    val centeresMaxEntriesPerNode: Int = 25
    logger.info("Centers # of partitions: %d".format(centers.getNumPartitions))
    val centersPartitioner: STRPartitioner = new STRPartitioner(centersNumPartitions
      , centersSampleRate
      , centersDimension
      , centersTransferThreshold
      , centeresMaxEntriesPerNode
      , centers)
    centers = centers.partitionBy(centersPartitioner)
      .cache()
    logger.info("Centers # of partitions: %d".format(centers.getNumPartitions))
    logger.info("04.Indexing centers... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCenters))
    // 05.Getting disks...
    timer = System.currentTimeMillis()
    val disks = centers
      .map(_._2)
      .toDS()
      .distanceJoin(p1, Array("x", "y"), Array("x1", "y1"), (epsilon / 2) + precision)
      .groupBy("id", "x", "y")
      .agg(collect_list("id1").alias("ids"))
    disks.cache()
    val nDisks = disks.count()
    logger.info("05.Getting disks... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nDisks))
    // 06.Filtering less-than-mu disks...
    timer = System.currentTimeMillis()
    val pointCandidates = disks
      .filter( row => row.getList[Int](3).size() >= mu )
      .map{ d =>
        (new Point(Array(d.getDouble(1), d.getDouble(2)))
          , Candidate(d.getLong(0)
          , d.getDouble(1)
          , d.getDouble(2)
          , d.getList[Int](3).asScala.mkString(",")
        )
        )
      }
      .rdd
    pointCandidates.cache()
    val nPointCandidates = pointCandidates.count()
    logger.info("06.Filtering less-than-mu disks... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nPointCandidates))
    // 07.Indexing candidates...
    timer = System.currentTimeMillis()
    val candidatesPartitionSize: Int = conf.candidates()
    val candidatesNumPartitions: Int = Math.ceil(nPointCandidates / candidatesPartitionSize).toInt
    val candidatesSampleRate: Double = 0.05
    val candidatesDimension: Int = 2
    val candidatesTransferThreshold: Long = 800 * 1024 * 1024
    val candidatesMaxEntriesPerNode: Int = 25
    logger.info("Candidates # of partitions: %d".format(pointCandidates.getNumPartitions))
    val candidatesPartitioner: STRPartitioner = new STRPartitioner(candidatesNumPartitions
      , candidatesSampleRate
      , candidatesDimension
      , candidatesTransferThreshold
      , candidatesMaxEntriesPerNode
      , pointCandidates)
    val candidates = pointCandidates.partitionBy(candidatesPartitioner)
      .map(_._2)
      .cache()
    logger.info("Candidates # of partitions: %d".format(candidates.getNumPartitions))
    candidates.cache()
    val nCandidates = candidates.count()
    logger.info("07.Indexing candidates... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidates))

    ////////////////////////////////////////////////////////////////////////////////
    timer = System.currentTimeMillis()
    val MBRs = candidatesPartitioner.mbrBound
      .map{ mbr =>
        "%d;%s".format(mbr._2, makeString(mbr._1))
      }
    new java.io.PrintWriter("/tmp/Expansion_D%s_E%.1f_M%d_MBRs.txt".format(conf.dataset(), epsilon, mu)) {
      write(MBRs.mkString("\n"))
      close()
    }
    val expandedMBRs = candidatesPartitioner.mbrBound
      .map{ mbr =>
        val mins = new Point( Array(mbr._1.low.coord(0) - epsilon,
          mbr._1.low.coord(1) - epsilon) )
        val maxs = new Point( Array(mbr._1.high.coord(0) + epsilon,
          mbr._1.high.coord(1) + epsilon) )
        (MBR(mins, maxs), mbr._2, 1)
      }
    val expandedRTree = RTree(expandedMBRs, candidatesMaxEntriesPerNode)
    new java.io.PrintWriter("/tmp/Expansion_D%s_E%.1f_M%d_EMBRs.txt".format(conf.dataset(), epsilon, mu)) {
      write(expandedMBRs.map(expandedMBR => "%d;%s".format(expandedMBR._2, makeString(expandedMBR._1))).mkString("\n"))
      close()
    }
    val candidatesByMBRId = pointCandidates.map{ candidate =>
      val candidateByMBRId = expandedRTree.circleRange(candidate._1, 0.0)
        .map{ t =>
          "%d;%f;%f;%d;%s".format(
            t._2
            , candidate._2.x
            , candidate._2.y
            , candidate._2.id
            , candidate._2.items
          )
        }
        .mkString("\n")
      candidateByMBRId
    }.cache()
    val nCandidatesByMBRId = candidatesByMBRId.count()
    logger.info("Getting expansions... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidatesByMBRId))
    new java.io.PrintWriter("/tmp/Expansion_D%s_E%.1f_M%d_Centers.txt".format(conf.dataset(), epsilon, mu)) {
      write(candidatesByMBRId.collect().mkString("\n"))
      close()
    }
    ////////////////////////////////////////////////////////////////////////////////

    /*
    ////////////////////////////////////////////////////////////////
    val mbrs = candidates.mapPartitionsWithIndex{ (index, iterator) =>
        var min_x: Double = Double.MaxValue
        var min_y: Double = Double.MaxValue
        var max_x: Double = Double.MinValue
        var max_y: Double = Double.MinValue

        var size: Int = 0

        iterator.toList.foreach{ row =>
            val x = row.x
            val y = row.y
            if(x < min_x){ min_x = x }
            if(y < min_y){ min_y = y }
            if(x > max_x){ max_x = x }
            if(y > max_y){ max_y = y }
            size += 1
        }
        List("%d;%d;%s".format(index, size, toWKT(min_x,min_y,max_x,max_y))).iterator
    }
    new java.io.PrintWriter("/tmp/MBRs_D%s_E%.1f_M%d.txt".format(conf.dataset(), epsilon, mu)) {
        write(mbrs.collect().mkString("\n"))
        close()
    }
    ////////////////////////////////////////////////////////////////

    // 08.Filtering candidate disks inside partitions...
    timer = System.currentTimeMillis()
    val candidatesInside = candidates
        //.rdd
        .mapPartitions { partition =>
            val pList = partition.toList
            val bbox = getBoundingBox(pList)
            val inside = pList
                .map( candidate => (candidate, isInside(candidate.x, candidate.y, bbox, epsilon)) )
                .filter(_._2)
                .map(_._1)
            inside.toIterator
        }
    candidatesInside.cache()
    val nCandidatesInside = candidatesInside.count()
    logger.info("08.Filtering candidate disks inside partitions... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidatesInside))
    // 09.Finding maximal disks inside partitions...
    timer = System.currentTimeMillis()
    val maximalsInside = candidatesInside
        .mapPartitionsWithIndex{ (index, partition) =>
            /*
            val transactions = partition
                .map { candidate =>
                    candidate.items
                    .split(",")
                    .map(new Integer(_))
                    .sorted.toList.asJava
                }.toList.asJava
            */

            ////////////////////////////////////////////////////////
            val transactions = partition
                .map { candidate =>
                    "%f;%f;%s".format(candidate.x, candidate.y, candidate.items)
                }.toList
            transactions.map(t => "%d;%d;%s".format(index, partition.size, t)).toIterator
            ////////////////////////////////////////////////////////

            //val algorithm = new AlgoFPMax
            //val maximals = algorithm.runAlgorithm(transactions, 1)
            //maximals.getItemsets(mu).asScala.toIterator
            //val LCM = new AlgoLCM
            //val data = new Transactions(transactions)
            //val closed = LCM.runAlgorithm(1, data)
            //val MFI = new AlgoCharmLCM
            //val maximals = MFI.runAlgorithm(closed)
            //maximals.getItemsets(mu).asScala.toIterator
        }
    maximalsInside.cache()
    val nMaximalsInside = maximalsInside.count()
    logger.info("09.Finding maximal disks inside partitions... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nMaximalsInside))

    ////////////////////////////////////////////////////////////////
    new java.io.PrintWriter("/tmp/PInside_D%s_E%.1f_M%d.txt".format(conf.dataset(), epsilon, mu)) {
        write(maximalsInside.collect().mkString("\n"))
        close()
    }
    ////////////////////////////////////////////////////////////////

    // 10.Filtering candidate disks on frame partitions...
    timer = System.currentTimeMillis()
    val candidatesFrame = candidates
        //.rdd
        .mapPartitions { partition =>
            val pList = partition.toList
            val bbox = getBoundingBox(pList)
            val frame = pList
                .map( candidate => (candidate, !isInside(candidate.x, candidate.y, bbox, epsilon)) )
                .filter(_._2)
                .map(_._1)
            frame.toIterator
        }
    val nCandidatesFrame = candidatesFrame.count()
    logger.info("10.Filtering candidate disks in frame partitions... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidatesFrame))
    // 11.Finding maximal disks in frame partitions...
    timer = System.currentTimeMillis()
    val maximalsFrame = candidatesFrame
        .mapPartitionsWithIndex { (index, partition) =>
            /*
            val transactions = partition
                .map { candidate =>
                    candidate.items
                        .split(",")
                        .map(new Integer(_))
                        .sorted.toList.asJava
                }.toSet.asJava
            */

            ////////////////////////////////////////////////////////
            val transactions = partition
                .map { candidate =>
                    "%f;%f;%s".format(candidate.x, candidate.y, candidate.items)
                }.toList
            transactions.map(t => "%d;%d;%s".format(index, partition.size, t)).toIterator
            ////////////////////////////////////////////////////////

            //val algorithm = new AlgoFPMax
            //val maximals = algorithm.runAlgorithm(transactions, 1)
            //maximals.getItemsets(mu).asScala.toIterator
            //val LCM = new AlgoLCM
            //val data = new Transactions(transactions)
            //val closed = LCM.runAlgorithm(1, data)
            //val MFI = new AlgoCharmLCM
            //val maximals = MFI.runAlgorithm(closed)
            //maximals.getItemsets(mu).asScala.toIterator
        }
    maximalsFrame.cache()
    val nMaximalsFrame = maximalsFrame.count()
    logger.info("11.Finding maximal disks in frame partitions... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nMaximalsFrame))

    ////////////////////////////////////////////////////////////////
    new java.io.PrintWriter("/tmp/PInframe_D%s_E%.1f_M%d.txt".format(conf.dataset(), epsilon, mu)) {
  write(maximalsFrame.collect().mkString("\n"))
  close()
}
    ////////////////////////////////////////////////////////////////

    // 12.Prunning redundants...
    timer = System.currentTimeMillis()
    val maximals = maximalsInside.union(maximalsFrame).distinct()
    maximals.cache()
    val nMaximals = maximals.count()
    logger.info("12.Prunning redundants... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nMaximals))
    val endTime = System.currentTimeMillis()
    val totalTime = (endTime - startTime)/1000.0
    // Printing info summary ...
    logger.info("%12s,%6s,%6s,%7s,%8s,%10s,%13s,%10s,%10s,%11s".
        format("Dataset", "Eps", "Cores", "Time",
            "# Pairs", "# Disks", "# Candidates",
            "# Inside", "# Inframe", "# Maximals"
        )
    )
    logger.info("%12s,%6.1f,%6d,%7.2f,%8d,%10d,%13d,%10d,%10d,%11d".
        format( conf.dataset(), conf.epsilon(), conf.cores(), totalTime,
            nPairs, nDisks, nCandidates,
            nMaximalsInside, nMaximalsFrame, nMaximals
        )
    )
    */
    // Dropping indices...
    timer = System.currentTimeMillis()
    p1.dropIndex()
    p2.dropIndex()
    logger.info("Dropping indices...[%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
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

  def isInside(x: Double, y: Double, bbox: BBox, epsilon: Double): Boolean ={
    x < (bbox.maxx - epsilon) &&
      x > (bbox.minx + epsilon) &&
      y < (bbox.maxy - epsilon) &&
      y > (bbox.miny + epsilon)
  }

  def getBoundingBox(c: List[Candidate]): BBox = {
    var minx: Double = Double.MaxValue
    var miny: Double = Double.MaxValue
    var maxx: Double = Double.MinValue
    var maxy: Double = Double.MinValue
    c.foreach{ d =>
      if(d.x < minx){ minx = d.x }
      if(d.x > maxx){ maxx = d.x }
      if(d.y < miny){ miny = d.y }
      if(d.y > maxy){ maxy = d.y }
    }
    BBox(minx, miny, maxx, maxy)
  }

  def toWKT(minx: Double, miny: Double, maxx: Double, maxy: Double): String = "POLYGON (( %f %f, %f %f, %f %f, %f %f, %f %f ))".
    format(
      minx, maxy,
      maxx, maxy,
      maxx, miny,
      minx, miny,
      minx, maxy
    )

  def makeString(mbr: MBR): String = toWKT(mbr.low.coord(0),mbr.low.coord(1),mbr.high.coord(0),mbr.high.coord(1))

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val epsilon:    ScallopOption[Double]   = opt[Double](default = Some(10.0))
    val mu:         ScallopOption[Int]      = opt[Int]   (default = Some(5))
    val entries:	ScallopOption[Int]      = opt[Int]   (default = Some(25))
    val partitions:	ScallopOption[Int]      = opt[Int]   (default = Some(1024))
    val centers:	ScallopOption[Int]      = opt[Int]   (default = Some(256))
    val candidates:	ScallopOption[Int]      = opt[Int]   (default = Some(64))
    val cores:      ScallopOption[Int]      = opt[Int]   (default = Some(28))
    val master:     ScallopOption[String]	= opt[String](default = Some("spark://169.235.27.138:7077")) /* spark://169.235.27.138:7077 */
    val path:       ScallopOption[String]	= opt[String](default = Some("Y3Q1/Datasets/"))
    val dataset:	ScallopOption[String]	= opt[String](default = Some("B20K"))
    val extension:	ScallopOption[String]	= opt[String](default = Some("csv"))
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
      .appName("MaximalFinder2")
      .config("simba.index.partitions",conf.partitions().toString)
      .config("spark.cores.max",conf.cores().toString)
      .getOrCreate()
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
    MaximalFinderExpansionDebugger.run(points, simba, conf)
    val end = System.currentTimeMillis()
    logger.info("Finishing MaximalFinder at %s...".format(DateTime.now.toLocalTime.toString))
    logger.info("Total time for MaximalFinder: %.3fms...".format((end - start)/1000.0))
    // Closing session...
    timer = System.currentTimeMillis()
    simba.close
    logger.info("Closing session... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
  }
}

