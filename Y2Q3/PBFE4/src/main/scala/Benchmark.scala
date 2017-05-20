package main.scala

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.Partitioner
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index.RTreeType
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.functions._
import org.apache.spark.rdd.PairRDDFunctions

/**
  * Created by and on 5/4/17.
  */
object Benchmark {

  case class APoint(id: Long, x: Double, y: Double)

  case class ACenter(id: Long, x: Double, y: Double)

  //case class Disk(cid: Long, pid: Int, IDs: java.util.List[Integer])
  case class Disk(id: Long, IDs: String)

  class CustomPartitioner(numParts: Int) extends Partitioner {
    override def numPartitions: Int = numParts

    override def getPartition(key: Any): Int = {
      val out = key.asInstanceOf[Long] >> 33
      out.toInt
    }
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

  def findDisks(pairsRDD: RDD[Row], epsilon: Double): RDD[ACenter] = {
    val r2: Double = math.pow(epsilon / 2, 2)
    val centers = pairsRDD
      .filter((row: Row) => row.getLong(0) != row.getLong(3))
      .map { (row: Row) =>
        val p1 = APoint(row.getLong(0), row.getDouble(1), row.getDouble(2))
        val p2 = APoint(row.getLong(3), row.getDouble(4), row.getDouble(5))
        calculateDiskCenterCoordinates(p1, p2, r2)
      }
    centers
  }

  def main(args: Array[String]): Unit = {
    // Setting some parameters...
    val MASTER: String = "local[*]"
    val LOGS: String = "ERROR"
    val MU: Int = 3
    val DSTART: Int = 10
    val DEND: Int = 10
    val DSTEP: Int = 10
    val ESTART: Double = 10.0
    val EEND: Double = 10.0
    val ESTEP: Double = 10.0
    val DELTA: Double = 0.01
    val POINT_SCHEMA = ScalaReflection.schemaFor[APoint].dataType.asInstanceOf[StructType]
    var PARTITIONS: Int = 32
    // Starting a session
    val simba = SimbaSession
      .builder()
      .master(MASTER)
      .appName("Benchmark")
      .config("simba.index.partitions", s"$PARTITIONS")
      .getOrCreate()
    simba.sparkContext.setLogLevel(LOGS)
    // Calling implicits
    import simba.simbaImplicits._
    import simba.implicits._
    // Looping with different datasets and epsilon values...
    for (dataset <- DSTART to DEND by DSTEP; epsilon <- ESTART to EEND by ESTEP) {
      val filename = s"/opt/Datasets/Beijing/P${dataset}K.csv"
      val tag = filename.substring(filename.lastIndexOf("/") + 1).split("\\.")(0).substring(1)
      // Reading the data...
      val points = simba.read
        .option("header", "true")
        .schema(POINT_SCHEMA)
        .csv(filename)
        .as[APoint]
      // Starting timer...
      val time1 = System.currentTimeMillis()
      // Indexing points...
      val p1 = points.toDF("id1", "x1", "y1")
      p1.index(RTreeType, "p1RT", Array("x1", "y1"))
      val p2 = points.toDF("id2", "x2", "y2")
      p2.index(RTreeType, "p2RT", Array("x2", "y2"))
      // Self-joining to find pairs of points close enough (< epsilon)...
      val pairsRDD = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon).rdd
      // Calculating disk's centers coordinates...
      val centers = findDisks(pairsRDD, epsilon).distinct()
        .toDS()
        .index(RTreeType, "centersRT", Array("x", "y"))
        .withColumn("id", monotonically_increasing_id())
        .as[ACenter]
      centers.cache()
      PARTITIONS = centers.rdd.getNumPartitions
      // Grouping objects enclosed by candidates disks...
      val candidates = centers
        .distanceJoin(p1, Array("x", "y"), Array("x1", "y1"), (epsilon / 2) + DELTA)
        .groupBy("id")
        .agg(collect_list("id1").alias("IDs"))
        // Filtering candidates less than mu...
        .filter(row => row.getList(1).size() >= MU)
        .map(d => (d.getLong(0), d.getList(1).toString))

      // Filtering redundant candidates

//      val n = candidates.count()
//      // Stopping timer...
//      val time2 = System.currentTimeMillis()
//      val time = (time2 - time1) / 1000.0
//      println(s"PFlock,$epsilon,$tag,$n,$time")
//      centers.show(10)
//      centers.rdd.mapPartitionsWithIndex((i, p) => Array(s"$i=${p.length}").toIterator).foreach(println)
//      println("Break")
//      candidates.show(10)
      val c = new PairRDDFunctions(candidates.rdd)
      c.partitionBy(new CustomPartitioner(numParts = PARTITIONS))
        .mapPartitionsWithIndex((i, p) => Array(s"$i=${p.length}").toIterator)
        .foreach(println)

    }
  }
}
