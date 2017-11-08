import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter, PrintWriter}

import JLCM.{ListCollector, TransactionsReader}
import Misc.GeoGSON
import SPMF.{AlgoCharmLCM, AlgoLCM, AlgoFPMax, Transactions}
import fr.liglab.jlcm.PLCM
import fr.liglab.jlcm.internals.ExplorationStep
import org.apache.spark.rdd.{DoubleRDDFunctions, RDD}
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.functions._
import org.apache.spark.sql.simba.index._
import org.apache.spark.sql.simba.{Dataset, SimbaSession}
import org.apache.spark.sql.types.StructType
import org.joda.time.DateTime
import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

/*******************************
*                              *
*  Created by and on 3/20/17.  *
*                              *
********************************/

object MaximalFinder {
	private val logger: Logger = LoggerFactory.getLogger("myLogger")
	
	case class SP_Point(id: Int, x: Double, y: Double)
	case class ACenter(id: Long, x: Double, y: Double)
	case class Candidate(id: Long, x: Double, y: Double, items: String)	
	case class BBox(minx: Double, miny: Double, maxx: Double, maxy: Double)

	var ESTART: Double = 10.0
	var EEND: Double = 10.0
	var ESTEP: Double = 10.0
	var MU: Int = 3
	var DATASET: String = "Berlin"
	var CORES: Int = 0
	var PARTITIONS: Int = 0
	var ENTRIES: Int = 25
	var LOG: List[String] = List("")
	var OUTPUT: List[String] = List.empty[String]
	private val PRECISION: Double = 0.001
	
	def run(points: Dataset[SP_Point]
			, timestamp: Int
			, simba: SimbaSession): RDD[List[Int]] = {
		var timer = System.currentTimeMillis()
		import simba.implicits._
		import simba.simbaImplicits._
		// Getting number of points...
		val n = points.count() 
		logger.info("Getting %d points... [%.3fms]".format(n, (System.currentTimeMillis() - timer)/1000.0))
		// Indexing...
		timer = System.currentTimeMillis()
		val p1 = points.toDF("id1", "x1", "y1")
		p1.index(RTreeType, "p1RT", Array("x1", "y1"))
		val p2 = points.toDF("id2", "x2", "y2")
		p2.index(RTreeType, "p2RT", Array("x2", "y2"))
		logger.info("Indexing... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
		// Setting final containers...
		var maximals: RDD[List[Int]] = simba.sparkContext.emptyRDD
		var nmaximals: Long = 0
		for( epsilon <- ESTART to EEND by ESTEP ){
			logger.info("Running epsilon = %.1f iteration...".format(epsilon))
			val startTime = System.currentTimeMillis()
			// Self-join...
			timer = System.currentTimeMillis()
			val pairsRDD = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon).
				filter((row: Row) => row.getInt(0) < row.getInt(3)).
				rdd
			pairsRDD.cache()
			val npairs = pairsRDD.count()
			logger.info("Self-join... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
			// Computing disks...
			timer = System.currentTimeMillis()
			val centers = findDisks(pairsRDD, epsilon).
				distinct().
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
			logger.info("MU=%d".format(MU))
			simba.sparkContext.broadcast(MU)
			val filteredCandidates = candidates.
				filter{ row => 
					
					row.getList[Integer](3).size() >= MU
				}.
				map(d => (d.getLong(0), d.getDouble(1), d.getDouble(2), d.getList[Integer](3).asScala.mkString(",")))
			val nFilteredCandidates = filteredCandidates.count()
			logger.info("Filtering less-than-mu disks... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
			// Indexing candidates...
			timer = System.currentTimeMillis()
			filteredCandidates.index(RTreeType, "candidatesRT", Array("_2", "_3"))
			filteredCandidates.cache()
			logger.info("Indexing candidates... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
			// Finding maximal disks inside partitions...
			var time1 = System.currentTimeMillis()
			val maximalsInside = filteredCandidates.
				rdd.
				mapPartitions { partition =>
					val transactions = partition.
						map { disk =>
							disk._4.
							split(",").
							map { id =>
								new Integer(id.trim)
							}.
							sorted.toList.asJava
						}.toList.asJava
					val fpMax = new AlgoFPMax
					val itemsets = fpMax.runAlgorithm(transactions, 1)
					itemsets.getItemsets(MU).asScala.toIterator
					//val LCM = new AlgoLCM
					//val data = new Transactions(transactions)
					//val closed = LCM.runAlgorithm(1, data)
					//val MFI = new AlgoCharmLCM
					//val maximals = MFI.runAlgorithm(closed)
					//maximals.getItemsets(MU).asScala.toIterator
					//closed.getMaximalItemsets1(MU).asScala.toIterator
				}
			maximalsInside.cache()
			val nMaximalsInside = maximalsInside.count()
			logger.info("Finding maximal disks inside partitions... [%.3fms]".format((System.currentTimeMillis() - time1)/1000.0))
			var time2 = System.currentTimeMillis()
			val timeI = (time2 - time1) / 1000.0
			// Filtering candidate disks on frame partitions...
			time1 = System.currentTimeMillis()
			val candidatesFrame = filteredCandidates.
				rdd.
				mapPartitions { partition =>
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
				}.
				toDS()
			candidatesFrame.cache()
			logger.info("Filtering candidate disks on frame partitions... [%.3fms]".format((System.currentTimeMillis() - time1)/1000.0))
			timer = System.currentTimeMillis()
			candidatesFrame.index(RTreeType, "candidatesFrameRT", Array("x", "y"))
			logger.info("Re-indexing candidate disks in frame partitions... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
			timer = System.currentTimeMillis()
			val maximalsFrame = candidatesFrame.
				rdd.
				mapPartitions { partition =>
					val transactions = partition.
						map { disk =>
							disk.items.
							split(",").
							map { id =>
								new Integer(id.trim)
							}.
							sorted.toList.asJava
						}.toList.asJava
					val fpMax = new AlgoFPMax
					val itemsets = fpMax.runAlgorithm(transactions, 1)
					itemsets.getItemsets(MU).asScala.toIterator
					//val LCM = new AlgoLCM
					//val data = new Transactions(transactions)
					//val closed = LCM.runAlgorithm(1, data)
					//val MFI = new AlgoCharmLCM
					//val maximals = MFI.runAlgorithm(closed)
					//maximals.getItemsets(MU).asScala.toIterator
					//closed.getMaximalItemsets1(MU).asScala.toIterator
				}
			val nMaximalsFrame = maximalsFrame.count()
			logger.info("Finding maximal disks in frame partitions... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
			time2 = System.currentTimeMillis()
			val timeF = (time2 - time1) / 1000.0
			// Prunning duplicates...
			timer = System.currentTimeMillis()
			maximals = maximalsInside.union(maximalsFrame).distinct().map(_.asScala.toList.map(_.intValue()))
			nmaximals = maximals.count()
			var endTime = System.currentTimeMillis()
			val totalTime = (endTime - startTime) / 1000.0
			logger.info("Prunning duplicates... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))

			///////////////////////////////////////////////////////////
			val outputFile = "/tmp/MaximalDisks_PFlocks_BeforeFinalLCM.csv"
			val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile)))
			maximalsInside.union(maximalsFrame).
				map(_.asScala.toList.map(_.intValue())).
				distinct().
				map{ pattern =>
					"%s\n".format(pattern.mkString(" "))
				}.
				collect().
				foreach(writer.write)
			writer.close()
			///////////////////////////////////////////////////////////

			// Printing info summary ...
			logger.info("%6s,%8s,%6s,%6s,%6s,%5s,%4s,%6s,%3s,%8s,%8s,%8s".
				format("Data", "# Cands", "# In", "# Fr", "# Max",
					"Part", "Ent", "Eps", "Mu", "TimeI", "TimeF", "Time"
				)
			)
			logger.info("%6s,%8d,%6d,%6d,%6d,%5d,%4d,%6.1f,%3d,%8.2f,%8.2f,%8.2f".
				format(DATASET, nFilteredCandidates, nMaximalsInside, nMaximalsFrame, nmaximals,
					PARTITIONS, ENTRIES, epsilon, MU, timeI, timeF, totalTime
				)
			)
			// Saving info summary to write on disk...
			OUTPUT = OUTPUT :+ s"PFLOCK,$epsilon,$MU,$timestamp,$timeI,$timeF,$totalTime,$ncandidates,$nmaximals,$CORES,$PARTITIONS,${org.joda.time.DateTime.now.toLocalTime}\n"
			// Dropping center and candidate indices...
			timer = System.currentTimeMillis()
			centers.dropIndexByName("centersRT")
			filteredCandidates.dropIndexByName("candidatesRT")
			candidatesFrame.dropIndexByName("candidatesFrameRT")
			logger.info("Dropping center and candidate indices... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
		}
		// Dropping point indices...
		timer = System.currentTimeMillis()
		p1.dropIndexByName("p1RT")
		p2.dropIndexByName("p2RT")
		logger.info("Dropping point indices... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))

		maximals
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

	def calculateDiskCenterCoordinates(p1: SP_Point, p2: SP_Point, r2: Double): ACenter = {
		val X: Double = p1.x - p2.x
		val Y: Double = p1.y - p2.y
		val D2: Double = math.pow(X, 2) + math.pow(Y, 2)
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

	def saveOutput(): Unit ={
		val outputFile = "%s_E%.1f%.1f-_M%d_C%d_P%d_%d.csv".format(DATASET, ESTART, EEND, MU, CORES, PARTITIONS, System.currentTimeMillis())
		val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile)))
		OUTPUT.foreach(writer.write)
		writer.close()
	}

	def drawGeoJSONs(args: Array[String]): Unit = {
		val DATASET = "B5K_Tester"
		val ENTRIES = "10"
		val PARTITIONS = "10"
		val EPSILON = 20.0
		val MU = 5
		val MASTER = "local[10]"
		val CORES = "10"
//		val DATASET = args(0)
//		val ENTRIES = args(1)
//		val PARTITIONS = args(2)
//		val EPSILON = args(3).toDouble
//		val MU = args(4).toInt
//		val MASTER = args(5)
//		val CORES = args(6)
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
		p2.index(RTreeType, "p2RT", Array("x2", "y2"))
		// Self-join...
		logger.info("Self-join...")
		val pairsRDD = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), EPSILON).rdd
		// Computing disks...
		logger.info("Computing disks...")
		val centers = findDisks(pairsRDD, EPSILON)
			.distinct()
			.toDS()
			.index(RTreeType, "centersRT", Array("x", "y"))
			.withColumn("id", monotonically_increasing_id())
			.as[ACenter]
		// Mapping disks and points...
		logger.info("Mapping disks and points...")
		val candidates = centers
			.distanceJoin(p1, Array("x", "y"), Array("x1", "y1"), (EPSILON / 2) + DELTA)
			.groupBy("id", "x", "y")
			.agg(collect_list("id1").alias("IDs"))
		val nCandidates = candidates.count()
		// Filtering less-than-mu disks...
		logger.info("Filtering less-than-mu disks...")
		val filteredCandidates = candidates.
			filter(row => row.getList(3).size() >= MU).
			map(d => (d.getLong(0), d.getDouble(1), d.getDouble(2), d.getList[Integer](3).asScala.mkString(",")))
		val nFilteredCandidates = filteredCandidates.count()
		// Setting final containers...
		var maximals: RDD[List[Int]] = simba.sparkContext.emptyRDD
		var nmaximals: Long = 0
		// Candidate indexing...
		logger.info("Candidate indexing...")
		filteredCandidates.index(RTreeType, "candidatesRT", Array("_2", "_3"))
		// Getting maximal disks inside partitions...
		logger.info("Getting maximal disks inside partitions...")
		var time1 = System.currentTimeMillis()
		val maximalsInside = filteredCandidates
			.rdd
			.mapPartitions { partition =>
				val transactions = partition.
					map { disk =>
						disk._4.
						split(",").
						map { id =>
							new Integer(id.trim)
						}.
						sorted.toList.asJava
					}.toSet.asJava
				val LCM = new AlgoLCM
				val data = new Transactions(transactions)
				val closed = LCM.runAlgorithm(1, data)
				closed.getMaximalItemsets1(MU).asScala.toIterator
			}
		val nMaximalsInside = maximalsInside.count()
		var time2 = System.currentTimeMillis()
		val timeI = (time2 - time1) / 1000.0
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
						(disk._1, disk._2, disk._3, disk._4, !isInside(disk._2, disk._3, bbox, EPSILON))
					}.
					filter(_._5).
					map { disk =>
						Candidate(disk._1, disk._2, disk._3, disk._4)
					}
				frame.toIterator
			}.toDS()
		candidatesFrame.count()
		logger.info("Re-indexing candidate disks in frames...")
		candidatesFrame.index(RTreeType, "candidatesFrameRT", Array("x", "y"))
		val maximalsFrame = candidatesFrame.rdd
			.mapPartitions { partition =>
				val transactions = partition.
					map { disk =>
						disk.items.
						split(",").
						map { id =>
							new Integer(id.trim)
						}.
						sorted.toList.asJava
					}.toSet.asJava
					val LCM = new AlgoLCM
					val data = new Transactions(transactions)
					val closed = LCM.runAlgorithm(1, data)
					closed.getMaximalItemsets1(MU).asScala.toIterator
			}
		val nMaximalsFrame = maximalsFrame.count()
		time2 = System.currentTimeMillis()
		val timeF = (time2 - time1) / 1000.0
		// Prunning duplicates...
		logger.info("Prunning duplicates...")
		maximals = maximalsInside.union(maximalsFrame).distinct().map(_.asScala.toList.map(_.intValue()))
		nmaximals = maximals.count()
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
		logger.info("%6s,%8s,%5s,%5s,%6s,%5s,%5s,%4s,%5s,%3s,%8s,%8s,%8s,%5s,%5s,%4s,%4s".
			format("Inside", "# Cands", "# In", "# Out", "# Max", 
				"Par1", "Par2", "Ent", "Eps", "Mu", 
				"TimeIn", "TimeOut", "Avg", "SD", "Var", "Min", "Max"
			)
		)
		logger.info("%6s,%8d,%5d,%5d,%6d,%5s,%5d,%4s,%5.1f,%3d,%8.2f,%8.2f,%8.2f,%5.2f,%5.2f,%4d,%4d".
			format(DATASET, nFilteredCandidates, nMaximalsInside, nMaximalsFrame, nmaximals,
				PARTITIONS, new_partitions, ENTRIES, EPSILON, MU,
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
		logger.info("%6s,%8s,%5s,%5s,%6s,%5s,%5s,%4s,%5s,%3s,%8s,%8s,%8s,%5s,%5s,%4s,%4s".
			format("Frame", "# Cands", "# In", "# Out", "# Max",
				"Par1", "Par2", "Ent", "Eps", "Mu", 
				"TimeIn", "TimeOut", "Avg", "SD", "Var", "Min", "Max"
			)
		)
		logger.info("%6s,%8d,%5d,%5d,%6d,%5s,%5d,%4s,%5.1f,%3d,%8.2f,%8.2f,%8.2f,%5.2f,%5.2f,%4d,%4d".
			format(DATASET, nFilteredCandidates, nMaximalsInside, nMaximalsFrame, nmaximals,
				PARTITIONS, new_partitions, ENTRIES, EPSILON, MU,
				timeI, timeF, avg, sd, variance, min, max
			)
		)
		// Writing Geojsons...
		logger.info("Writing Geojsons...")
		val mbrs = filteredCandidates.rdd.mapPartitionsWithIndex{ (index, iterator) =>
			var min_x: Double = Double.MaxValue
			var min_y: Double = Double.MaxValue
			var max_x: Double = Double.MinValue
			var max_y: Double = Double.MinValue
			var size: Int = 0
			iterator.toList.foreach{ row =>
				val x = row._2
				val y = row._3
				if(x < min_x){ min_x = x }
				if(y < min_y){ min_y = y }
				if(x > max_x){ max_x = x }
				if(y > max_y){ max_y = y }
				size += 1
			}
			List((min_x, min_y, max_x, max_y, index, size)).iterator
		}
		val gson1 = new GeoGSON(EPSG)
		mbrs.collect().foreach{ row =>
			gson1.makeMBRs(row._1, row._2, row._3, row._4, row._5, row._6)
		}
		var geojson: String = "%s%sViz/RTree_%s_E%.1f_M%d_P%s.geojson".format(phd_home, path, DATASET, EPSILON, MU, PARTITIONS)
		gson1.saveGeoJSON(geojson)
		logger.info("%s has been written...".format(geojson))
		val gson2 = new GeoGSON(EPSG)
		filteredCandidates.collect().foreach{ row =>
			gson2.makePoints(row._2, row._3)
		}
		geojson = "%s%sViz/Points_%s_E%.1f_M%d_P%s.geojson".format(phd_home, path, DATASET, EPSILON, MU, PARTITIONS)
		gson2.saveGeoJSON(geojson)
		logger.info("%s has been written...".format(geojson))
		// Dropping indices...
		logger.info("Dropping indices...")
		centers.dropIndexByName("centersRT")
		filteredCandidates.dropIndexByName("candidatesRT")		
		candidatesFrame.dropIndexByName("candidatesFrameRT")
		p1.dropIndexByName("p1RT")
		p2.dropIndexByName("p2RT")
		// Closing app...
		logger.info("Closing app...")
		simba.stop()
	}

	class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
		val estart:	ScallopOption[Double]	= opt[Double](default = Some(10.0))
		val eend:	ScallopOption[Double]	= opt[Double](default = Some(10.0))
		val estep:	ScallopOption[Double]	= opt[Double](default = Some(10.0))
		val mu:		ScallopOption[Int]	= opt[Int]   (default = Some(5))
		val entries:	ScallopOption[Int]	= opt[Int]   (default = Some(25))
		val partitions:	ScallopOption[Int]	= opt[Int]   (default = Some(256))
		val cores:	ScallopOption[Int]	= opt[Int]   (default = Some(3))
		val master:	ScallopOption[String]	= opt[String](default = Some("local[*]"))
		val path:	ScallopOption[String]	= opt[String](default = Some("Y3Q1/Datasets/"))
		val dataset:	ScallopOption[String]	= opt[String](default = Some("B20K"))
		val extension:	ScallopOption[String]	= opt[String](default = Some("csv"))
		verify()
	}
	
	def main(args: Array[String]): Unit = {
		// Starting app...
		logger.info("Starting app at %s...".format(DateTime.now.toLocalDateTime.toString))
		// Reading arguments from command line...
		var timer = System.currentTimeMillis()
		val conf = new Conf(args)
		// Setting global variables...
		MaximalFinder.DATASET = conf.dataset()
		MaximalFinder.ENTRIES = conf.entries()
		MaximalFinder.PARTITIONS = conf.partitions()
		MaximalFinder.ESTART = conf.estart()
		MaximalFinder.EEND = conf.eend()
		MaximalFinder.ESTEP = conf.estep()
		MaximalFinder.MU = conf.mu()
		MaximalFinder.CORES = conf.cores()
		// Setting local variables...
		val POINT_SCHEMA = ScalaReflection.schemaFor[SP_Point].dataType.asInstanceOf[StructType]
		val MASTER = conf.master()
		logger.info("Setting variables... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
		// Starting session...
		timer = System.currentTimeMillis()
		val simba = SimbaSession.builder().
			master(MASTER).
			appName("MaximalFinder").
			config("simba.rtree.maxEntriesPerNode", MaximalFinder.ENTRIES).
			config("simba.index.partitions", "1024").
			config("spark.cores.max", MaximalFinder.CORES).
			getOrCreate()
		import simba.implicits._
		import simba.simbaImplicits._
		logger.info("Starting session... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
       	// Reading...
		timer = System.currentTimeMillis()
		val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
		val filename = "%s%s%s.%s".format(phd_home, conf.path(), MaximalFinder.DATASET, conf.extension())
		val points = simba.read.option("header", "false").schema(POINT_SCHEMA).csv(filename).as[SP_Point]
		logger.info("Reading %s... [%.3fms]".format(filename, (System.currentTimeMillis() - timer)/1000.0))
		// Running MaximalFinder...
		logger.info("Lauching MaximalFinder at %s...".format(DateTime.now.toLocalTime.toString))
		timer = System.currentTimeMillis()
		MaximalFinder.run(points, 0, simba)
		logger.info("Finishing MaximalFinder at %s...]".format(DateTime.now.toLocalTime.toString))
		logger.info("Total time for MaximalFinder: %.3fms...".format((System.currentTimeMillis() - timer)/1000.0))
		// Closing session...
		timer = System.currentTimeMillis()
		simba.close
		logger.info("Closing session... [%.3fms]".format((System.currentTimeMillis() - timer)/1000.0))
		// Ending app...
		logger.info("Ending app at %s...".format(DateTime.now.toLocalDateTime.toString))
	}	
}
