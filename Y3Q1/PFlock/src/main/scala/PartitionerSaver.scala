import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}
import scala.collection.JavaConverters._
import org.apache.spark.rdd.DoubleRDDFunctions
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index._
import org.slf4j.{Logger, LoggerFactory}

/**
  * Created by and on 3/20/17.
  */

object PartitionSaver {
	private val logger: Logger = LoggerFactory.getLogger("myLogger")
	
	case class SP_Point(id: Int, x: Double, y: Double)
    case class APoint(id: Int, x: Double, y: Double)
	case class ACenter(id: Long, x: Double, y: Double)
	case class Candidate(id: Long, x: Double, y: Double, items: String)
	case class BBox(minx: Double, miny: Double, maxx: Double, maxy: Double)

	def main(args: Array[String]): Unit = {
		val dataset = args(0)
		val epsilon = args(1).toDouble
		val mu = args(2).toInt
		val cores = args(3)
		val partitions = args(4)
		val master = "spark://169.235.27.134:7077"
		val POINT_SCHEMA = ScalaReflection.schemaFor[SP_Point].dataType.asInstanceOf[StructType]
		val CANDIDATE_SCHEMA = ScalaReflection.schemaFor[Candidate].dataType.asInstanceOf[StructType]
		val DELTA = 0.01
		// Setting session...
		logger.info("Setting session...")
      val simba = SimbaSession.builder().
				master(master).
				appName("PartitionerSaver").
				config("simba.index.partitions", partitions).
				config("spark.cores.max", cores).
				getOrCreate()
		import simba.implicits._
		import simba.simbaImplicits._
    // Reading...
    var timer = System.currentTimeMillis()
		val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
		val path = "Y3Q1/Datasets/"
		val extension = "csv"
		val filename = "%s%s%s.%s".format(phd_home, path, dataset, extension)
		val points = simba.read.option("header", "false").schema(POINT_SCHEMA).csv(filename).as[SP_Point].cache()
		val nPoints = points.count()
		logger.info("Reading... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nPoints))
		var pointsNumPartitions = points.rdd.getNumPartitions
		logger.info("points,Before indexing,%d".format(pointsNumPartitions))
    // Indexing points...
    timer = System.currentTimeMillis()
    points.createOrReplaceTempView("points")
    simba.indexTable("points", RTreeType, "pointsRT",  Array("x", "y") )
    simba.showIndex("points")
		val p1 = simba.sql("SELECT id AS id1, x AS x1, y AS y1 FROM points")
		val p2 = simba.sql("SELECT id AS id2, x AS x2, y AS y2 FROM points")
    logger.info("Indexing points... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nPoints))
		val p1NumPartitions = points.rdd.getNumPartitions
		logger.info("p1,After indexing,%d".format(p1NumPartitions))
		val p2NumPartitions = points.rdd.getNumPartitions
		logger.info("p2,After indexing,%d".format(p2NumPartitions))
    // Getting pairs...
    timer = System.currentTimeMillis()
		val pairs = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon).rdd.cache()
		val nPairs = pairs.count()
    logger.info("Getting pairs... [%.3fs] [%d results]".format((System.currentTimeMillis() - timer)/1000.0, nPairs))
		/*
		// Computing disks...
		logger.info("Computing disks...")
		val centers = findDisks(pairsRDD, epsilon)
			.distinct()
			.toDS()
			.index(RTreeType, "centersRT", Array("x", "y"))
			.withColumn("id", monotonically_increasing_id())
			.as[ACenter]
		// Mapping disks and points...
		logger.info("Mapping disks and points...")
		val candidates = centers
			.distanceJoin(p1, Array("x", "y"), Array("x1", "y1"), (epsilon / 2) + DELTA)
			.groupBy("id", "x", "y")
			.agg(collect_list("id1").alias("IDs"))
		val ncandidates = candidates.count()
		// Filtering less-than-mu disks...
		logger.info("Filtering less-than-mu disks...")
		val filteredCandidates = candidates.
			filter(row => row.getList(3).size() >= mu).
			map(d => (d.getLong(0), d.getDouble(1), d.getDouble(2), d.getList[Integer](3).asScala.mkString(",")))
		val nFilteredCandidates = filteredCandidates.count()
		// Candidate indexing...
		logger.info("Candidate indexing...")
		filteredCandidates.index(RTreeType, "candidatesRT", Array("_2", "_3"))
		// Getting maximal disks inside partitions...
		logger.info("Getting candidate disks inside partitions...")
		var time1 = System.currentTimeMillis()
		val candidatesInside = filteredCandidates
			.rdd
			.mapPartitionsWithIndex { (index, partition) =>
				val transactions = partition.
					map { disk =>
						disk._4.
						split(",").
						map { id =>
							new Integer(id.trim)
						}.
						sorted.toList.asJava
					}.toList.
					map { t =>
						"%d;%s\n".format(index, t.asScala.mkString(","))
					}
				transactions.toIterator
			}
		val nCandidatesInside = candidatesInside.count()
		var time2 = System.currentTimeMillis()
		val timeI = (time2 - time1) / 1000.0
		val candidatesInsideFile = "%s%sCandidatesInside_%s_E%.1f_M%d_P%s.csv".format(phd_home, path, dataset, epsilon, mu, partitions)
		val candidatesInsideWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(candidatesInsideFile)))
		candidatesInside.collect().foreach(candidatesInsideWriter.write)
		candidatesInsideWriter.close()
		logger.info("Writing candidate inside at %s...".format(candidatesInsideFile))
		// Getting maximal disks on frame partitions...
		logger.info("Getting maximal disks on frame partitions...")
		time1 = System.currentTimeMillis()
		val candidatesFrame = filteredCandidates
			.rdd
			.mapPartitions { partition =>
				val pList = partition.toList
				val bbox = getBoundingBox(pList)
				val frame = pList.
					map{ disk => 
						(disk._1, disk._2, disk._3, disk._4, !isInside(disk._2, disk._3, bbox, epsilon))
					}.
					filter(_._5).
					map { disk =>
						Candidate(disk._1, disk._2, disk._3, disk._4)
					}
				frame.toIterator
			}.toDS()
		val nCandidatesFrame = candidatesFrame.count()
		logger.info("Re-indexing candidate disks in frames...")
		candidatesFrame.index(RTreeType, "candidatesFrameRT", Array("x", "y"))
		val candidatesFrameReindexed = candidatesFrame.rdd
			.mapPartitionsWithIndex { (index, partition) =>
				val transactions = partition.
					map { disk =>
						disk.items.
						split(",").
						map { id =>
							new Integer(id.trim)
						}.
						sorted.toList.asJava
					}.toList.
					map { t =>
						"%d;%s\n".format(index, t.asScala.mkString(","))
					}
				transactions.toIterator
			}
		val ncandidatesFrameReindexed = candidatesFrameReindexed.count()
		time2 = System.currentTimeMillis()
		val timeF = (time2 - time1) / 1000.0
		val candidatesFrameReindexedFile = "%s%sCandidatesFrame_%s_E%.1f_M%d_P%s.csv".format(phd_home, path, dataset, epsilon, mu, partitions)
		val candidatesFrameReindexedWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(candidatesFrameReindexedFile)))
		candidatesFrameReindexed.collect().foreach(candidatesFrameReindexedWriter.write)
		candidatesFrameReindexedWriter.close()
		logger.info("Writing candidate frame at %s...".format(candidatesFrameReindexedFile))
		// Collecting candidate disks inside stats...
		logger.info("Collecting candidate disks inside stats...")
		var partition_sizes = filteredCandidates.rdd.
			mapPartitions{ it =>
				List(it.size.toDouble).iterator
			}
		var new_partitions = filteredCandidates.rdd.getNumPartitions
		var sizes = new DoubleRDDFunctions(partition_sizes)
		var avg = sizes.mean()
		var sd = sizes.stdev()
		var variance = sizes.variance()
		var max = partition_sizes.max().toInt
		var min = partition_sizes.min().toInt
		// Printing candidate disks inside summary ...
		logger.info("%6s,%8s,%5s,%5s,%5s,%5s,%4s,%5s,%3s,%8s,%8s,%8s,%5s,%5s,%4s,%4s".
			format("Inside", "# Cands", "# In", "# Out", 
				"Par1", "Par2", "Ent", "Eps", "Mu", 
				"TimeIn", "TimeOut", "Avg", "SD", "Var", "Min", "Max"
			)
		)
		logger.info("%6s,%8d,%5d,%5d,%5s,%5d,%4s,%5.1f,%3d,%8.2f,%8.2f,%8.2f,%5.2f,%5.2f,%4d,%4d".
			format(dataset, nFilteredCandidates, nCandidatesInside, ncandidatesFrameReindexed,
				partitions, new_partitions, entries, epsilon, mu, 
				timeI, timeF, avg, sd, variance, min, max
			)
		)
		// Collecting candidate disks frame stats...
		logger.info("Collecting candidate disks frame stats...")
		partition_sizes = candidatesFrame.rdd.
			mapPartitions{ it =>
				List(it.size.toDouble).iterator
			}
		new_partitions = candidatesFrame.rdd.getNumPartitions
		sizes = new DoubleRDDFunctions(partition_sizes)
		avg = sizes.mean()
		sd = sizes.stdev()
		variance = sizes.variance()
		max = partition_sizes.max().toInt
		min = partition_sizes.min().toInt
		// Printing candidate disks frame summary ...
		logger.info("%6s,%8s,%5s,%5s,%5s,%5s,%4s,%5s,%3s,%8s,%8s,%8s,%5s,%5s,%4s,%4s".
			format("Frame", "# Cands", "# In", "# Out", 
				"Par1", "Par2", "Ent", "Eps", "Mu", 
				"TimeIn", "TimeOut", "Avg", "SD", "Var", "Min", "Max"
			)
		)
		logger.info("%6s,%8d,%5d,%5d,%5s,%5d,%4s,%5.1f,%3d,%8.2f,%8.2f,%8.2f,%5.2f,%5.2f,%4d,%4d".
			format(dataset, nFilteredCandidates, nCandidatesInside, ncandidatesFrameReindexed,
				partitions, new_partitions, entries, epsilon, mu, 
				timeI, timeF, avg, sd, variance, min, max
			)
		)
		*/
		// Dropping indices...
		logger.info("Dropping indices...")
		//centers.dropIndexByName("centersRT")
		//filteredCandidates.dropIndexByName("candidatesRT")
		//candidatesFrame.dropIndexByName("candidatesFrameRT")
    p1.dropIndex()
    p2.dropIndex()
		// Closing app...
		logger.info("Closing app...")
		simba.stop()
	}
	
	def findDisks(pairsRDD: RDD[Row], epsilon: Double): RDD[ACenter] = {
		val r2: Double = math.pow(epsilon / 2, 2)
		val centers = pairsRDD
			.filter((row: Row) => row.getInt(0) != row.getInt(3))
			.map { (row: Row) =>
				val p1 = APoint(row.getInt(0), row.getDouble(1), row.getDouble(2))
				val p2 = APoint(row.getInt(3), row.getDouble(4), row.getDouble(5))
				calculateDiskCenterCoordinates(p1, p2, r2)
			}
		centers
	}

	def calculateDiskCenterCoordinates(p1: APoint, p2: APoint, r2: Double): ACenter = {
		val X: Double = p1.x - p2.x
		val Y: Double = p1.y - p2.y
		var D2: Double = math.pow(X, 2) + math.pow(Y, 2)
		if (D2 == 0)
			D2 = 0.01
		val root: Double = math.sqrt(math.abs(4.0 * (r2 / D2) - 1.0))
		val h1: Double = ((X + Y * root) / 2) + p2.x
		val k1: Double = ((Y - X * root) / 2) + p2.y

		ACenter(0, h1, k1)
	}

	def isInside(x: Double, y: Double, bbox: BBox, epsilon: Double): Boolean ={
		x < (bbox.maxx - epsilon) &&
		x > (bbox.minx + epsilon) &&
		y < (bbox.maxy - epsilon) &&
		y > (bbox.miny + epsilon)
	}

	def getBoundingBox(p: List[(Long, Double, Double, Any)]): BBox = {
		var minx: Double = Double.MaxValue
		var miny: Double = Double.MaxValue
		var maxx: Double = Double.MinValue
		var maxy: Double = Double.MinValue
		p.foreach{r =>
			if(r._2 < minx){ minx = r._2 }
			if(r._2 > maxx){ maxx = r._2 }
			if(r._3 < miny){ miny = r._3 }
			if(r._3 > maxy){ maxy = r._3 }
		}
		BBox(minx, miny, maxx, maxy)
	}	
}
