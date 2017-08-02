import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}

import org.apache.spark.Partitioner
import org.apache.spark.rdd.{PairRDDFunctions, RDD}
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.functions._
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index.RTreeType
import org.apache.spark.sql.types.StructType

import scala.collection.JavaConverters._
import SPMF.AlgoFPMax
import org.rogach.scallop._


/**
  * Created by and on 5/4/17.
  */
object PFlock {

  def main(args: Array[String]): Unit = {
    var times = List.empty[String] 
    times = times :+ s"[{'content':'Starting app...','start':'${org.joda.time.DateTime.now.toLocalDate} ${org.joda.time.DateTime.now.toLocalTime}'},\n"
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
    val simba = SimbaSession
      .builder()
      .master(MASTER)
      .appName("PFlock")
      .config("simba.index.partitions", s"${conf.partitions()}")
      .config("spark.cores.max", s"${conf.cores()}")
      .config("spark.eventLog.enabled","true")
      .config("spark.eventLog.dir", s"file://${conf.dirlogs()}")
      .getOrCreate()
    println(s"Running ${simba.sparkContext.applicationId} on ${conf.cores()} cores...")
    println("Tag  \tEpsilon\tDataset\tN\tTimeD\tTimeM\tCores\tTimestamp")
    times = times :+ s"{'content':'Setting paramaters...','start':'${org.joda.time.DateTime.now.toLocalDate} ${org.joda.time.DateTime.now.toLocalTime}'},\n"

    simba.sparkContext.setLogLevel(conf.logs())
    // Calling implicits...
    import simba.implicits._
    import simba.simbaImplicits._
    //import scala.collection.JavaConverters._
    var output = List.empty[String]
    // Looping with different datasets and epsilon values...
    for (dataset <- conf.dstart() to conf.dend() by conf.dstep();
         epsilon <- conf.estart() to conf.eend() by conf.estep()) {
      val filename = s"${conf.prefix()}$dataset${conf.suffix()}.csv"
      val tag = filename.substring(filename.lastIndexOf("/") + 1).split("\\.")(0).substring(1)
      // Reading data...
      val points = simba.read
        .option("header", "true")
        .schema(POINT_SCHEMA)
        .csv(filename)
        .as[APoint]
      // Starting timer...
      times = times :+ s"{'content':'Reading data...','start':'${org.joda.time.DateTime.now.toLocalDate} ${org.joda.time.DateTime.now.toLocalTime}'},\n"
      var time1 = System.currentTimeMillis()
      // Indexing points...
      val p1 = points.toDF("id1", "x1", "y1")
      p1.index(RTreeType, "p1RT", Array("x1", "y1"))
      val p2 = points.toDF("id2", "x2", "y2")
      p2.index(RTreeType, "p2RT", Array("x2", "y2"))
      times = times :+ s"{'content':'Indexing points...','start':'${org.joda.time.DateTime.now.toLocalDate} ${org.joda.time.DateTime.now.toLocalTime}'},\n"
      // Self-joining to find pairs of points close enough (< epsilon)...
      val pairsRDD = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon).rdd
      times = times :+ s"{'content':'Finding pairs (Self-join)...','start':'${org.joda.time.DateTime.now.toLocalDate} ${org.joda.time.DateTime.now.toLocalTime}'},\n"
      // Calculating disk's centers coordinates...
      val centers = findDisks(pairsRDD, epsilon).distinct()
        .toDS()
        .index(RTreeType, "centersRT", Array("x", "y"))
        .withColumn("id", monotonically_increasing_id())
        .as[ACenter]
      times = times :+ s"{'content':'Computing disks...','start':'${org.joda.time.DateTime.now.toLocalDate} ${org.joda.time.DateTime.now.toLocalTime}'},\n"
      centers.cache()
      times = times :+ s"{'content':'Caching...','start':'${org.joda.time.DateTime.now.toLocalDate} ${org.joda.time.DateTime.now.toLocalTime}'},\n"
      val PARTITIONS: Int = centers.rdd.getNumPartitions
      val MU: Int = conf.mu()
      // Grouping objects enclosed by candidates disks...
      val disksAndPoints = centers
        .distanceJoin(p1, Array("x", "y"), Array("x1", "y1"), (epsilon / 2) + conf.delta())
        .groupBy("id", "x", "y")
        .agg(collect_list("id1").alias("IDs"))
      times = times :+ s"{'content':'Maping disks and points...','start':'${org.joda.time.DateTime.now.toLocalDate} ${org.joda.time.DateTime.now.toLocalTime}'},\n"
      // Filtering candidates less than mu...
      val candidatesPair =  disksAndPoints.filter(row => row.getList(3).size() >= MU)
        .rdd
        .map(d => (d.getLong(0), (d.getDouble(1), d.getDouble(2), d.getList[Integer](3))))
      times = times :+ s"{'content':'Filtering less-than-mu disks...','start':'${org.joda.time.DateTime.now.toLocalDate} ${org.joda.time.DateTime.now.toLocalTime}'},\n"
      var time2 = System.currentTimeMillis()
      val timeD = (time2 - time1) / 1000.0
      time1 = System.currentTimeMillis()
      // Filtering redundant candidates
      val candidates = new PairRDDFunctions(candidatesPair)
      val c = candidates.partitionBy(new CustomPartitioner(numParts = PARTITIONS))
        .mapPartitions{ partition =>
          val pList = partition.toList
          val bbox = getBoundingBox(pList)
          val disks = pList.map(disk => (disk._2, isInBuffer(disk._2, bbox, epsilon)))
          //val localList = disks.filter(_._2).map(_._1._3).asJava
          //val fpMax = new AlgoFPMax
          //val itemsets = fpMax.runAlgorithm(localList, 1)
          //itemsets.getItemsets(MU).asScala.toIterator
          disks.toIterator
        }
      times = times :+ s"{'content':'Filtering redundants (Not full implemented)...','start':'${org.joda.time.DateTime.now.toLocalDate} ${org.joda.time.DateTime.now.toLocalTime}'},\n"
      val n = c.count()
      times = times :+ s"{'content':'Final counting...','start':'${org.joda.time.DateTime.now.toLocalDate} ${org.joda.time.DateTime.now.toLocalTime}'},\n"
      // Stopping timer...
      time2 = System.currentTimeMillis()
      val timeM = (time2 - time1) / 1000.0
      
      // Print summary...
      val record = s"PFlock,$epsilon,$tag,$n,$timeD,$timeM,${conf.cores()},${org.joda.time.DateTime.now.toLocalTime}\n"
      output = output :+ record
      print(record.replaceAll(",", "\t"))
      
      // Dropping indices
      p1.dropIndexByName("p1RT")
      p2.dropIndexByName("p2RT")
      times = times :+ s"{'content':'Dropping indices...','start':'${org.joda.time.DateTime.now.toLocalDate} ${org.joda.time.DateTime.now.toLocalTime}'},\n"
    }
    val filename = s"${conf.output()}_N${conf.dstart()}${conf.suffix()}-${conf.dend()}${conf.suffix()}_E${conf.estart()}-${conf.eend()}_C${conf.cores()}_${conf.tag()}.csv"
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename)))
    output.foreach(writer.write)
    writer.close()
    simba.close()
    times = times :+ s"{'content':'Closing app...','start':'${org.joda.time.DateTime.now.toLocalDate} ${org.joda.time.DateTime.now.toLocalTime}'}]"
    val json = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("times.json")))
    times.foreach(json.write)
    json.close()
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
    val D2: Double = math.pow(X, 2) + math.pow(Y, 2)
    if (D2 == 0)
      throw new UnsupportedOperationException("Identical points...")
    val root: Double = math.sqrt(math.abs(4.0 * (r2 / D2) - 1.0))
    val h1: Double = ((X + Y * root) / 2) + p2.x
    val k1: Double = ((Y - X * root) / 2) + p2.y

    ACenter(0, h1, k1)
  }

  def isInBuffer(tuple: (Double, Double, Any), bbox: BBox, epsilon: Double): Boolean ={
    val x = tuple._1
    val y = tuple._2
    x < bbox.maxx - epsilon &&
      x > bbox.minx + epsilon &&
      y < bbox.maxy - epsilon &&
      y > bbox.miny + epsilon
  }

  def getBoundingBox(p: List[(Long, (Double, Double, Any))]): BBox = {
    var minx: Double = Double.MaxValue
    var miny: Double = Double.MaxValue
    var maxx: Double = Double.MinValue
    var maxy: Double = Double.MinValue
    p.foreach{r =>
      if(r._2._1 < minx){
        minx = r._2._1
      }
      if (r._2._1 > maxx){
        maxx = r._2._1
      }
      if(r._2._2 < miny){
        miny = r._2._2
      }
      if(r._2._2 > maxy){
        maxy = r._2._2
      }
    }
    BBox(minx, miny, maxx, maxy)
  }

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
    val estep: ScallopOption[Double] = opt[Double](default = Some(2.0))
    val delta: ScallopOption[Double] = opt[Double](default = Some(0.01))
    val partitions: ScallopOption[Int] = opt[Int](default = Some(16))
    val cores: ScallopOption[Int] = opt[Int](default = Some(4))
    val master: ScallopOption[String] = opt[String](default = Some("local[*]"))
    val logs: ScallopOption[String] = opt[String](default = Some("ERROR"))
    val output: ScallopOption[String] = opt[String](default = Some("output"))
    val prefix: ScallopOption[String] = opt[String](default = Some("/opt/Datasets/Beijing/P"))
    val suffix: ScallopOption[String] = opt[String](default = Some("K"))
    val dirlogs: ScallopOption[String] = opt[String](default = Some("/opt/Spark/Logs"))
    val tag: ScallopOption[String] = opt[String](default = Some(""))

    verify()
  }
}
