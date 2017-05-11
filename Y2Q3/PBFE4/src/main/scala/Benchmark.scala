package main.scala

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index.RTreeType
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.catalyst.ScalaReflection

/**
  * Created by and on 5/4/17.
  */
object Benchmark {

  case class APoint(id: Long, x: Double, y: Double)
  case class ACenter(id: Long, x: Double, y: Double)

  class Test1(epsilon: Double) extends Serializable {

    def calculateDisksLeft(p1: APoint, p2: APoint, r2: Double): ACenter = {
      val X: Double = p1.x - p2.x
      val Y: Double = p1.y - p2.y
      val D2: Double = math.pow(X, 2) + math.pow(Y, 2)
      if (D2 == 0)
        throw new UnsupportedOperationException("Identical points...")
      val root: Double = math.sqrt(math.abs(4.0 * (r2 / D2) - 1.0))
      val h1: Double = ((X + Y * root) / 2) + p2.x
      //val h2: Double = ((X - Y * root) / 2) + pair.getFloat(4)
      val k1: Double = ((Y - X * root) / 2) + p2.y
      //val k2: Double = ((Y + X * root) / 2) + pair.getFloat(5)

      ACenter(0, h1, k1)
    }

    def run(pairsRDD: RDD[Row]): RDD[ACenter] = {
      val r2: Double = math.pow(epsilon / 2, 2)
      val centers = pairsRDD
        .filter((row: Row) => row.getLong(0) != row.getLong(3))
        .map { (row: Row) =>
          val p1 = APoint(row.getLong(0), row.getDouble(1), row.getDouble(2))
          val p2 = APoint(row.getLong(3), row.getDouble(4), row.getDouble(5))
          calculateDisksLeft(p1, p2, r2)
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
      .config("simba.index.partitions", "16")
      .getOrCreate()
    simba.sparkContext.setLogLevel(logs)

    import simba.simbaImplicits._
    import simba.implicits._

    val schema = ScalaReflection.schemaFor[APoint].dataType.asInstanceOf[StructType]
    for (dataset <- 10 to 10 by 10; epsilon <- 10.0 to 10.0 by 10.0) {
      val filename = s"/opt/Datasets/Beijing/P${dataset}K.csv"
      val tag = filename.substring(filename.lastIndexOf("/") + 1).split("\\.")(0).substring(1)

      val points = simba.read.
        option("header", "true").
        schema(schema).
        csv(filename).
        as[APoint]

      val p1 = points.toDF("id1", "x1", "y1")
      p1.index(RTreeType, "p1RT", Array("x1", "y1"))
      val p2 = points.toDF("id2", "x2", "y2")
      p2.index(RTreeType, "p2RT", Array("x2", "y2"))

      val time1 = System.currentTimeMillis()
      val test1 = new Test1(epsilon)
      val pairsRDD = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon).rdd
      val centers = test1.run(pairsRDD).distinct().toDS()
      centers.index(RTreeType, "centersRT", Array("x", "y"))
      val c = centers.rdd.mapPartitionsWithIndex{(index, partition) =>
        val n = partition.size
        val e = s"${n}".size
        val start = index * Math.pow(10, e).toLong
        val cids = (start to start + n)

        (partition.toVector zip cids.toVector).toList.toIterator
      }

      c.foreach(println)
      //val n = centers.count()
      //val time2 = System.currentTimeMillis()
      //val time = (time2 - time1) / 1000.0
      //println(s"PFlock,$epsilon,$tag,$n,$time")
    }
    //centers.index(RTreeType, "centeresRT", Array("x", "y"))
  }
}
