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
//import SPMF.{AlgoCharmLCM, AlgoFPMax, AlgoLCM, Transactions}

import scala.collection.JavaConverters._

object MaximalFinderExpansion {
    private val logger: Logger = LoggerFactory.getLogger("myLogger")
    private val precision: Double = 0.001
    private val dimensions: Int = 2
    private val sampleRate: Double = 0.05
    private var nPoints: Long = 0

    case class SP_Point(id: Long, x: Double, y: Double)
    case class Center(id: Long, x: Double, y: Double)
    case class Pair(id1: Long, x1: Double, y1: Double, id2: Long, x2: Double, y2: Double)
    case class Candidate(id: Long, x: Double, y: Double, items: String)
    case class PointCandidate(pid: Long, cid: Long)
    case class BBox(minx: Double, miny: Double, maxx: Double, maxy: Double)

    def run(points: Dataset[SP_Point]
            , simba: SimbaSession
            , conf: Conf): Unit = {
        val mu = conf.mu()
        val epsilon = conf.epsilon()
        import simba.implicits._
        import simba.simbaImplicits._
        logger.info("00.Setting mu=%d,epsilon=%.1f,cores=%d,candidatesPartitions=%d,dataset=%s".
            format(mu, epsilon, conf.cores(), conf.candidates(), conf.dataset()))
        val startTime = System.currentTimeMillis()
        // 1.Indexing points...
        var timer = System.currentTimeMillis()
        val p1 = points.toDF("id1", "x1", "y1")
        p1.index(RTreeType, "p1RT", Array("x1", "y1"))
        val p2 = points.toDF("id2", "x2", "y2")
        p2.index(RTreeType, "p1RT", Array("x2", "y2"))
        logger.info("01.Indexing points... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nPoints))
        // 02.Getting pairs...
        timer = System.currentTimeMillis()
        val pairs = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon)
            .as[Pair]
            .filter(pair => pair.id1 < pair.id2)
            .rdd
            .cache()
        val nPairs = pairs.count()
        logger.info("02.Getting pairs... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nPairs))
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
        logger.info("03.Computing centers... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCenters))
        // 04.Indexing centers...
        timer = System.currentTimeMillis()
        centers.index(RTreeType, "centersRT", Array("x", "y")).cache()
        logger.info("04.Indexing centers... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCenters))
        // 05.Getting disks...
        timer = System.currentTimeMillis()
        val disks = centers
            .distanceJoin(p1, Array("x", "y"), Array("x1", "y1"), (epsilon / 2) + precision)
            .groupBy("id", "x", "y")
            .agg(collect_list("id1").alias("ids"))
            .cache()
        val nDisks = disks.count()
        logger.info("05.Getting disks... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nDisks))
        // 06.Filtering less-than-mu disks...
        timer = System.currentTimeMillis()
        val filteredDisks = disks
            .filter(row => row.getList[Int](3).size() >= mu)
            .rdd
            .cache()
        val nFilteredDisks = filteredDisks.count()
        logger.info("06.Filtering less-than-mu disks... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nFilteredDisks))
        // 07.Indexing candidates...
        timer = System.currentTimeMillis()
        var candidates = filteredDisks.map{ d =>
                (new Point(Array(d.getDouble(1), d.getDouble(2)))
                    , Candidate(d.getLong(0), d.getDouble(1), d.getDouble(2), d.getList[Int](3).asScala.mkString(","))
                )
            }
            .cache()
        val nCandidates = candidates.count()
        val candidatesSampleRate: Double = sampleRate
        val candidatesDimension: Int = dimensions
        val candidatesTransferThreshold: Long = 800 * 1024 * 1024
        val candidatesMaxEntriesPerNode: Int = 25
        val candidatesPartitionSize: Int = conf.candidates()
        var candidatesNumPartitions: Int = Math.ceil(nCandidates / candidatesPartitionSize).toInt
        logger.info("Candidates # of partitions: %d".format(candidates.getNumPartitions))
        val candidatesPartitioner: STRPartitioner = new STRPartitioner(candidatesNumPartitions
            , candidatesSampleRate
            , candidatesDimension
            , candidatesTransferThreshold
            , candidatesMaxEntriesPerNode
            , candidates)
        // Re-partition just for debugging purposes...
        candidates = candidates.partitionBy(candidatesPartitioner)
            .cache()
        candidatesNumPartitions = candidates.getNumPartitions
        logger.info("Candidates # of partitions: %d".format(candidatesNumPartitions))
        logger.info("07.Indexing candidates... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidates))
        // 08.Getting expansions...

        ////////////////////////////////////////////////////////////////
        saveStringArray(candidatesPartitioner.mbrBound.map(mbr => "%d;%s".format(mbr._2, mbr2wkt(mbr._1))), "MBRs", conf)
        ////////////////////////////////////////////////////////////////

        timer = System.currentTimeMillis()
        val expandedMBRs = candidatesPartitioner.mbrBound
            .map{ mbr =>
                val mins = new Point( Array(mbr._1.low.coord(0) - epsilon, mbr._1.low.coord(1) - epsilon) )
                val maxs = new Point( Array(mbr._1.high.coord(0) + epsilon, mbr._1.high.coord(1) + epsilon) )
                (MBR(mins, maxs), mbr._2, 1)
            }
        val expandedRTree = RTree(expandedMBRs, candidatesMaxEntriesPerNode)
        val candidatesByMBRId = candidates.flatMap{ candidate =>
            expandedRTree.circleRange(candidate._1, 0.0)
                .map(mbr => (mbr._2, candidate._2))
            }
            .partitionBy(new ExpansionPartitioner(candidatesNumPartitions))
            .distinct()
            .cache()
        val nCandidatesByMBRId = candidatesByMBRId.count()
        logger.info("08.Getting expansions... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidatesByMBRId))
        // 09.Tagging candidates in the expansion area...
        timer = System.currentTimeMillis()
        val pointCandidates = candidatesByMBRId.map{ c =>
            ( c._2.id, c._2.items.split(",").map(_.toLong).toList)
        }
        .toDF
        .withColumn("pid", explode($"items"))
        .select("cid","pid")
        .as[PointCandidate]
        .cache()
        pointCandidates.join(points, pointCandidates.col("pid") === points.col("id")).show()
        logger.info("08.09.Tagging candidates in the expansion area... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidatesByMBRId))
        // 09.Finding maximal disks...

        ////////////////////////////////////////////////////////////////
        saveStringArray(expandedMBRs.map(mbr => "%d;%s".format(mbr._2, mbr2wkt(mbr._1))), "EMBRs", conf)
        ////////////////////////////////////////////////////////////////

        timer = System.currentTimeMillis()
        var maximals = candidatesByMBRId 
            .mapPartitionsWithIndex{ (index, partition) =>
                /*
                val transactions = partition
                    .map { candidate =>
                        candidate._2.items
                        .split(",")
                        .map(new Integer(_))
                        .sorted.toList.asJava
                    }.toSet.asJava
                */

                ////////////////////////////////////////////////////////

                partition.map(t => "%d;%s".format(index, t))

                ////////////////////////////////////////////////////////

                //val algorithm = new AlgoFPMax
                //val maximals = algorithm.runAlgorithm(transactions, 1)
                //maximals.getItemsets(mu).asScala.toIterator
                //val LCM = new AlgoLCM
                //val data = new Transactions(transactions)
                //val closed = LCM.runAlgorithm(1, data)
                //closed.getMaximalItemsets1(mu).asScala.toIterator
                //val MFI = new AlgoCharmLCM
                //val maximals = MFI.runAlgorithm(closed)
                //maximals.getItemsets(mu).asScala.toIterator
            }
            .cache()
        var nMaximals = maximals.count()
        logger.info("09.Finding maximal disks... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nMaximals))

        ////////////////////////////////////////////////////////////////
        saveStringArray(maximals.collect(), "Maximals", conf)
        ////////////////////////////////////////////////////////////////

        // 10.Prunning duplicates...
        timer = System.currentTimeMillis()
        maximals = maximals.distinct().cache()
        nMaximals = maximals.count()
        logger.info("10.Prunning redundants... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nMaximals))
        val endTime = System.currentTimeMillis()
        val totalTime = (endTime - startTime)/1000.0
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
        // Dropping indices...
        timer = System.currentTimeMillis()
        p1.dropIndex()
        p2.dropIndex()
        centers.dropIndex()
        logger.info("Dropping indices...[%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
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

    def isInside(x: Double, y: Double, bbox: BBox, epsilon: Double): Boolean ={
        x < (bbox.maxx - epsilon) &&
            x > (bbox.minx + epsilon) &&
            y < (bbox.maxy - epsilon) &&
            y > (bbox.miny + epsilon)
    }

    def saveStringArray(array: Array[String], filename: String, conf: Conf): Unit = {
        new java.io.PrintWriter("/tmp/D%s_E%.1f_M%d_%s.txt".format(conf.dataset(), conf.epsilon(), conf.mu(), filename)) {
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
        val epsilon:    ScallopOption[Double]   = opt[Double](default = Some(10.0))
        val mu:         ScallopOption[Int]      = opt[Int]   (default = Some(5))
        val entries:	ScallopOption[Int]      = opt[Int]   (default = Some(25))
        val partitions:	ScallopOption[Int]      = opt[Int]   (default = Some(1024))
        val candidates:	ScallopOption[Int]      = opt[Int]   (default = Some(256))
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
            .appName("MaximalFinderExpansion")
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
        val points = simba.read
            .option("header", "false")
            .schema(POINT_SCHEMA)
            .csv(filename)
            .as[SP_Point]
            .cache()
        nPoints = points.count()
        logger.info("Reading dataset... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
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
        logger.info("Closing session... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
    }
}
