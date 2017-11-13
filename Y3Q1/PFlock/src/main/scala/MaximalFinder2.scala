import SPMF.{AlgoCharmLCM, AlgoFPMax, AlgoLCM, Transactions}
import scala.collection.JavaConverters._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.simba.{Dataset, SimbaSession}
import org.apache.spark.sql.simba.index.RTreeType
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType
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
        val mu = conf.mu()
        val epsilon = conf.epsilon()
        import simba.implicits._
        import simba.simbaImplicits._
        logger.info("00.Setting mu=%d,epsilon=%.1f,cores=%d,partitions=%d,dataset=%s".
            format(mu, epsilon, conf.cores(), conf.partitions(), conf.dataset()))
        val startTime = System.currentTimeMillis()
        // 1.Indexing points...
        var timer = System.currentTimeMillis()
        val p1 = points.toDF("id1", "x1", "y1")
        p1.index(RTreeType, "p1RT", Array("x1", "y1"))
        val p2 = points.toDF("id2", "x2", "y2")
        p2.index(RTreeType, "p2RT", Array("x2", "y2"))
        logger.info("01.Indexing points... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, n))

		////////////////////////////////////////////////////////////////
        logger.info("P1 # of partitions: %d".format(p1.rdd.getNumPartitions))
        logger.info("P2 # of partitions: %d".format(p2.rdd.getNumPartitions))
		////////////////////////////////////////////////////////////////

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
        val centers = leftCenters.union(rightCenters)
            .toDF("x","y")
        centers.cache()
        val nCenters = centers.count()
        logger.info("03.Computing centers... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCenters))
        // 04.Indexing centers...
        timer = System.currentTimeMillis()
        val centersIndexed = centers.index(RTreeType, "centersRT", Array("x", "y"))
			.withColumn("id", monotonically_increasing_id())
            .as[SP_Point]
        logger.info("04.Indexing centers... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCenters))

		////////////////////////////////////////////////////////////////
        logger.info("Centers # of partitions: %d".format(centersIndexed.rdd.getNumPartitions))
		////////////////////////////////////////////////////////////////

        // 05.Getting disks...
        timer = System.currentTimeMillis()
        val disks = centersIndexed
            .distanceJoin(p1, Array("x", "y"), Array("x1", "y1"), (epsilon / 2) + precision)
            .groupBy("id", "x", "y")
            .agg(collect_list("id1").alias("ids"))
        disks.cache()
        val nDisks = disks.count()
        logger.info("05.Getting disks... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nDisks))
        // 06.Filtering less-than-mu disks...
        timer = System.currentTimeMillis()
        var candidates = disks
            .filter( row => row.getList[Int](3).size() >= mu )
            .map(d => Candidate(d.getLong(0), d.getDouble(1), d.getDouble(2), d.getList[Int](3).asScala.mkString(",")))
        candidates.cache()
        val nCandidates = candidates.count()
        logger.info("06.Filtering less-than-mu disks... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidates))
        // 07.Indexing candidates...
        timer = System.currentTimeMillis()
        candidates.index(RTreeType, "candidatesRT", Array("x", "y"))
        logger.info("candidates # of partitions: %d".format(candidates.rdd.getNumPartitions))
        logger.info("07.Indexing candidates... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidates))
        // 08.Filtering candidate disks inside partitions...
        timer = System.currentTimeMillis()
        val candidatesInside = candidates
            .rdd
            .mapPartitions { partition =>
                val pList = partition.toList
                val bbox = getBoundingBox(pList)
                val frame = pList
                    .map( candidate => (candidate, isInside(candidate.x, candidate.y, bbox, epsilon)) )
                    .filter(_._2)
                    .map(_._1)
                frame.toIterator
            }
        candidatesInside.cache()
        val nCandidatesInside = candidatesInside.count()
        logger.info("08.Filtering candidate disks inside partitions... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidatesInside))
        // 09.Finding maximal disks inside partitions...
        timer = System.currentTimeMillis()
        val maximalsInside = candidatesInside
            .mapPartitionsWithIndex{ (index, partition) =>
                val transactions = partition
                    .map { candidate =>
                        candidate.items
                        .split(",")
                        .map(new Integer(_))
                        .sorted.toList.asJava
                    }.toList.asJava
                    
                ////////////////////////////////////////////////////////
                transactions.asScala.map(_.asScala.map(_.toInt))
					.map(t => "%d, %s".format(index, t.mkString(" "))).toIterator
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
        new java.io.PrintWriter("/home/acald013/PhD/Y3Q1/Validation/Inside_D%s_E%.1f_M%d.txt".format(conf.dataset(), epsilon, mu)) { 
			write(maximalsInside.collect().mkString("\n"))
			close 
		}
        ////////////////////////////////////////////////////////////////

        // 10.Filtering candidate disks on frame partitions...
        timer = System.currentTimeMillis()
        val candidatesFrame = candidates
            .rdd
            .mapPartitions { partition =>
                val pList = partition.toList
                val bbox = getBoundingBox(pList)
                val frame = pList
                    .map( candidate => (candidate, !isInside(candidate.x, candidate.y, bbox, epsilon)) )
                    .filter(_._2)
                    .map(_._1)
                frame.toIterator
            }.toDS()
        val nCandidatesFrame = candidatesFrame.count()
        logger.info("10.Filtering candidate disks in frame partitions... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidatesFrame))
        // 11.Re-indexing candidate disks in frame partitions
        timer = System.currentTimeMillis()
        candidatesFrame.index(RTreeType, "candidatesFrameRT", Array("x", "y"))
        logger.info("11.Re-indexing candidate disks in frame partitions... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nCandidatesFrame))
        // 12.Finding maximal disks in frame partitions...
        timer = System.currentTimeMillis()
        val maximalsFrame = candidatesFrame.rdd
            .mapPartitionsWithIndex { (index, partition) =>
                val transactions = partition
                    .map { candidate =>
                        candidate.items
                            .split(",")
                            .map(new Integer(_))
                            .sorted.toList.asJava
                    }.toList.asJava
                    
                ////////////////////////////////////////////////////////
                transactions.asScala.map(_.asScala.map(_.toInt))
					.map(t => "%d, %s".format(index, t.mkString(" "))).toIterator
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
        logger.info("12.Finding maximal disks in frame partitions... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nMaximalsFrame))

        ////////////////////////////////////////////////////////////////
        new java.io.PrintWriter("/home/acald013/PhD/Y3Q1/Validation/Inframe_D%s_E%.1f_M%d.txt".format(conf.dataset(), epsilon, mu)) { 
			write(maximalsFrame.collect().mkString("\n"))
			close 
		}
        ////////////////////////////////////////////////////////////////

        // 13.Prunning redundants...
        timer = System.currentTimeMillis()
        val maximals = maximalsInside.union(maximalsFrame).distinct()
        maximals.cache()
        val nMaximals = maximals.count()
        logger.info("13.Prunning redundants... [%.3fms] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nMaximals))
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
        // Dropping indices...
        timer = System.currentTimeMillis()
        p1.dropIndex()
        centers.dropIndex()
        candidates.dropIndex()
        candidatesFrame.dropIndex()
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

    class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
        val epsilon:	ScallopOption[Double]	= opt[Double](default = Some(10.0))
        val mu:		ScallopOption[Int]	= opt[Int]   (default = Some(5))
        val entries:	ScallopOption[Int]	= opt[Int]   (default = Some(25))
        val partitions:	ScallopOption[Int]	= opt[Int]   (default = Some(1024))
        val cores:	ScallopOption[Int]	= opt[Int]   (default = Some(3))
        val master:	ScallopOption[String]	= opt[String](default = Some("local[*]"))
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
        val simba = SimbaSession.builder().master("spark://169.235.27.138:7077").appName("MaximalFinder2").config("simba.index.partitions","1024").config("spark.cores.max","28").getOrCreate()
        import simba.implicits._
        import simba.simbaImplicits._
        logger.info("Starting session... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
        // Reading...
        timer = System.currentTimeMillis()
        val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
        //val filename = "%s%s%s.%s".format(phd_home, conf.path(), conf.dataset(), conf.extension())
        val filename = "%s%s%s.%s".format("/home/acald013/PhD/", "Y3Q1/Datasets/", "B60K_PFlock", "csv")
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
