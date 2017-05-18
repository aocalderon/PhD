package main.scala

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index.RTreeType
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.functions._

/**
  * Created by and on 5/4/17.
  */
object Benchmark {

  case class APoint(id: Long, x: Double, y: Double)
  case class ACenter(id: Long, x: Double, y: Double)

  class Test1(epsilon: Double) extends Serializable {

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

    def findDisks(pairsRDD: RDD[Row]): RDD[ACenter] = {
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
  }

  def main(args: Array[String]): Unit = {
    val master: String = "local[*]"
    //var filename: String = "/opt/Datasets/Beijing/P10K.csv"
    //val epsilon: Double = 100.0
    //val mu: Int = 3
    val logs: String = "ERROR"

    val simba = SimbaSession
      .builder()
      .master(master)
      .appName("Benchmark")
      .config("simba.index.partitions", "32")
      .getOrCreate()
    simba.sparkContext.setLogLevel(logs)

    import simba.simbaImplicits._
    import simba.implicits._

    // Setting some parameters...
    val DSTART: Int = 10
    val DEND: Int = 10
    val DSTEP: Int = 10
    val ESTART: Double = 10.0
    val EEND: Double = 10.0
    val ESTEP: Double = 10.0
    val DELTA: Double = 0.01
    val POINT_SCHEMA = ScalaReflection.schemaFor[APoint].dataType.asInstanceOf[StructType]

    for (dataset <- DSTART to DEND by DSTEP; epsilon <- ESTART to EEND by ESTEP) {
      val filename = s"/opt/Datasets/Beijing/P${dataset}K.csv"
      val tag = filename.substring(filename.lastIndexOf("/") + 1).split("\\.")(0).substring(1)

      val points = simba.read.
        option("header", "true").
        schema(POINT_SCHEMA).
        csv(filename).
        as[APoint]

      // Start timer...
      val time1 = System.currentTimeMillis()
      // Points indexing...
      val p1 = points.toDF("id1", "x1", "y1")
      p1.index(RTreeType, "p1RT", Array("x1", "y1"))
      val p2 = points.toDF("id2", "x2", "y2")
      p2.index(RTreeType, "p2RT", Array("x2", "y2"))
      // Self-join
      val test1 = new Test1(epsilon)
      val pairsRDD = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon).rdd
      // Calculate disk's centers coordinates...
      val centers_without_id = test1.findDisks(pairsRDD).distinct()
      // Adding ID to centers...
      val k = centers_without_id.mapPartitions( x => Seq(x.length).toIterator ).max()
      val e = s"$k".length
      val p = Math.pow(10, e).toLong
      val centers = centers_without_id.
        mapPartitionsWithIndex{(index, centers) =>
          val start = index * p
          val cids = (start to start + k).toIterator
          (centers zip cids).map{ case (center, cid) =>
            ACenter(cid, center.x, center.y)
          }.toList.toIterator
        }.toDS()
      centers.index(RTreeType, "centersRT", Array("x", "y"))
      centers.cache()
      val objectsByDisk = centers.
        distanceJoin(p1, Array("x", "y"), Array("x1", "y1"), (epsilon / 2) + DELTA).
        groupBy("id").
        agg(collect_list("id1"))
      val n = objectsByDisk.count()
      // Stop timer...
      val time2 = System.currentTimeMillis()
      val time = (time2 - time1) / 1000.0

      objectsByDisk.show(100)
      println(s"PFlock,$epsilon,$tag,$n,$time")
    }
    //centers.index(RTreeType, "centeresRT", Array("x", "y"))
  }
}
