import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}

import SPMF.AlgoFPMax
import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.functions._
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index.RTreeType
import org.apache.spark.sql.types.StructType
import org.rogach.scallop._
import scala.collection.JavaConverters._

/**
  * Created by and on 5/4/17.
  */
object PFlock {

  def main(args: Array[String]): Unit = {
    var times = List.empty[String] 
    times = times :+ s"""{"content":"Starting app...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
    // Reading arguments from command line...
    val conf = new Conf(args)
    // Tuning master and number of cores...
    var MASTER = conf.master()
    if(conf.cores() == 1){
      MASTER = "local[1]"
    }
    // Setting parameters...
    val POINT_SCHEMA = ScalaReflection.schemaFor[APoint].dataType.asInstanceOf[StructType]
    // Starting a session...
    times = times :+ s"""{"content":"Setting paramaters...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
    val simba = SimbaSession
      .builder()
      .master(MASTER)
      .appName("PFlock")
      .config("simba.index.partitions", s"${conf.partitions()}")
      .config("spark.cores.max", s"${conf.cores()}")
      //.config("spark.eventLog.enabled","true")
      //.config("spark.eventLog.dir", s"file://${conf.dirlogs()}")
      .getOrCreate()
    simba.sparkContext.setLogLevel(conf.logs())
    // Calling implicits...
    import simba.implicits._
    import simba.simbaImplicits._
    var output = List.empty[String]
    println(s"Running ${simba.sparkContext.applicationId} on ${conf.cores()} cores and ${conf.partitions()} partitions...")
    println("%10.10s %10.10s %10.10s %10.10s %10.10s %10.10s %10.10s %10.10s %10.10s %10.10s %15.15s"
      .format("Tag","Epsilon","Dataset","TimeD","TimeM","TotalTime","NCandidates","NMaximal","Cores", "Partitions","Timestamp"))
    // Looping with different datasets and epsilon values...
    for (dataset <- conf.dstart() to conf.dend() by conf.dstep();
         epsilon <- conf.estart() to conf.eend() by conf.estep()) {
      val filename = s"${conf.prefix()}$dataset${conf.suffix()}.csv"
      val tag = filename.substring(filename.lastIndexOf("/") + 1).split("\\.")(0).substring(1)
      // Reading data...
      times = times :+ s"""{"content":"Reading data...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
      val points = simba.read
        .option("header", "false")
        .schema(POINT_SCHEMA)
        .csv(filename)
        .as[APoint]
      // Starting timer...
      times = times :+ s"""{"content":"Indexing points...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
      var time1: Long = System.currentTimeMillis()
      // Indexing points...
      val p1 = points.toDF("id1", "x1", "y1")
      p1.index(RTreeType, "p1RT", Array("x1", "y1"))
      val p2 = points.toDF("id2", "x2", "y2")
      p2.index(RTreeType, "p2RT", Array("x2", "y2"))
      // Self-joining to find pairs of points close enough (< epsilon)...
      times = times :+ s"""{"content":"Finding pairs (Self-join)...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
      val pairsRDD = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon).rdd
      // Calculating disk's centers coordinates...
      times = times :+ s"""{"content":"Computing disks...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
      val centers = findDisks(pairsRDD, epsilon)
        .distinct()
        .toDS()
        .index(RTreeType, "centersRT", Array("x", "y"))
        .withColumn("id", monotonically_increasing_id())
        .as[ACenter]
      val PARTITIONS: Int = centers.rdd.getNumPartitions
      val MU: Int = conf.mu()
      // Grouping objects enclosed by candidates disks...
      times = times :+ s"""{"content":"Mapping disks and points...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
      val candidates = centers
        .distanceJoin(p1, Array("x", "y"), Array("x1", "y1"), (epsilon / 2) + conf.delta())
        .groupBy("id", "x", "y")
        .agg(collect_list("id1").alias("IDs"))
      val ncandidates = candidates.count()
      var time2: Long = System.currentTimeMillis()
      val timeD: Double = (time2 - time1) / 1000.0
      // Filtering candidates less than mu...
      time1 = System.currentTimeMillis()
      times = times :+ s"""{"content":"Filtering less-than-mu disks...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
      val filteredCandidates =  candidates.filter(row => row.getList(3).size() >= MU)
        .map(d => (d.getLong(0), d.getDouble(1), d.getDouble(2), d.getList[Integer](3).toString))
      var nmaximal: Long = 0
      // Prevent indexing of empty collections...
      if(filteredCandidates.count() != 0){
        // Indexing remaining candidates disks...
        filteredCandidates.index(RTreeType, "candidatesRT", Array("_2", "_3"))
        // Filtering redundant candidates
        times = times :+ s"""{"content":"Getting maximals inside...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
        val maximalInside = filteredCandidates
          .rdd
          .mapPartitions { partition =>
            val transactions = partition
              .map { disk =>
                disk._4
                  .replace("[","")
                  .replace("]","")
                  .split(",")
                  .map{ id =>
                    new Integer(id.trim)
                  }
                  .sorted
                  .toList
                  .asJava
              }.toList.asJava
            val fpMax = new AlgoFPMax
            val itemsets = fpMax.runAlgorithm(transactions, 1)
            itemsets.getItemsets(MU).asScala.toIterator
          }
        maximalInside.count()
        times = times :+ s"""{"content":"Getting maximals in frame...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
        val maximalFrame = filteredCandidates
          .rdd
          .mapPartitions { partition =>
            val pList = partition.toList
            val bbox = getBoundingBox(pList)
            val transactions = pList
              .map(disk => (disk._1, disk._2, disk._3, disk._4, !isInside(disk._2, disk._3, bbox, epsilon)))
              .filter(_._5)
              .map { disk =>
                disk._4
                  .replace("[","")
                  .replace("]","")
                  .split(",")
                  .map{ id =>
                    new Integer(id.trim)
                  }
                  .sorted
                  .toList
                  .asJava
              }.asJava
            val fpMax = new AlgoFPMax
            val itemsets = fpMax.runAlgorithm(transactions, 1)
            itemsets.getItemsets(MU).asScala.toIterator
          }
        maximalFrame.count()
        times = times :+ s"""{"content":"Prunning duplicates...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
        val maximal = maximalInside.union(maximalFrame).distinct()
        nmaximal = maximal.count()
      }
      // Stopping timer...
      time2 = System.currentTimeMillis()
      val timeM: Double = (time2 - time1) / 1000.0
      val time: Double = BigDecimal(timeD + timeM).setScale(3, BigDecimal.RoundingMode.HALF_DOWN).toDouble
      // Print summary...
      val record = s"PFlock,$epsilon,$tag,$timeD,$timeM,$time,$ncandidates,$nmaximal,${conf.cores()},$PARTITIONS,${org.joda.time.DateTime.now.toLocalTime}\n"
      output = output :+ record
      print("%10.10s %10.1f %10.10s %10.3f %10.3f %10.3f %10d %10d %10d %10d %15.15s\n"
        .format("PFlock",epsilon,tag,timeD,timeM,time,ncandidates,nmaximal,conf.cores(),PARTITIONS,org.joda.time.DateTime.now.toLocalTime))

      // Dropping indices
      times = times :+ s"""{"content":"Dropping indices...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
      p1.dropIndexByName("p1RT")
      p2.dropIndexByName("p2RT")
      centers.dropIndexByName("centersRT")
      filteredCandidates.dropIndexByName("candidatesRT")
    }
    val filename = s"${conf.output()}_N${conf.dstart()}${conf.suffix()}-${conf.dend()}${conf.suffix()}_E${conf.estart()}-${conf.eend()}_C${conf.cores()}_M${conf.mu()}_P${conf.partitions()}_${conf.tag()}.csv"
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename)))
    output.foreach(writer.write)
    writer.close()
    val jsonname = s"${conf.dirlogs()}/${simba.sparkContext.applicationId}.json"
    val json = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(jsonname)))
    times.foreach(json.write)
    json.close()
    times = times :+ s"""{"content":"Closing app...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"}"""
    simba.close()
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
      if(r._2 < minx){
        minx = r._2
      }
      if (r._2 > maxx){
        maxx = r._2
      }
      if(r._3 < miny){
        miny = r._3
      }
      if(r._3 > maxy){
        maxy = r._3
      }
    }
    BBox(minx, miny, maxx, maxy)
  }

  def toWKT(bbox: BBox): String = "POLYGON (( %f %f, %f %f, %f %f, %f %f, %f %f ))"
    .format(
      bbox.minx, bbox.maxy,
      bbox.maxx, bbox.maxy,
      bbox.maxx, bbox.miny,
      bbox.minx, bbox.miny,
      bbox.minx, bbox.maxy
    )

  def toWKT(x: Double, y: Double): String = "POINT (%f %f)".format(x, y)

  case class APoint(id: Int, x: Double, y: Double)

  case class ACenter(id: Long, x: Double, y: Double)

  case class BBox(minx: Double, miny: Double, maxx: Double, maxy: Double)

  class CustomPartitioner(numParts: Int) extends Partitioner {
    override def numPartitions: Int = numParts

    override def getPartition(key: Any): Int = {
      val out = key.asInstanceOf[Long] >> 33
      out.toInt
    }
  }

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val mu: ScallopOption[Int] = opt[Int](default = Some(3))
    val dstart: ScallopOption[Int] = opt[Int](default = Some(10))
    val dend: ScallopOption[Int] = opt[Int](default = Some(10))
    val dstep: ScallopOption[Int] = opt[Int](default = Some(10))
    val estart: ScallopOption[Double] = opt[Double](default = Some(10.0))
    val eend: ScallopOption[Double] = opt[Double](default = Some(10.0))
    val estep: ScallopOption[Double] = opt[Double](default = Some(10.0))
    val delta: ScallopOption[Double] = opt[Double](default = Some(0.01))
    val partitions: ScallopOption[Int] = opt[Int](default = Some(32))
    val cores: ScallopOption[Int] = opt[Int](default = Some(4))
    val master: ScallopOption[String] = opt[String](default = Some("local[*]"))
    val logs: ScallopOption[String] = opt[String](default = Some("ERROR"))
    val output: ScallopOption[String] = opt[String](default = Some("output"))
    val prefix: ScallopOption[String] = opt[String](default = Some("/opt/Datasets/Berlin/B"))
    val suffix: ScallopOption[String] = opt[String](default = Some("K_3068"))
    val dirlogs: ScallopOption[String] = opt[String](default = Some("/opt/Spark/Logs"))
    val tag: ScallopOption[String] = opt[String](default = Some(""))

    verify()
  }
}
