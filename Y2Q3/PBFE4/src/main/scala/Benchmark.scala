package main.scala

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index.RTreeType

/**
  * Created by and on 5/4/17.
  */
object Benchmark {

  case class Apoint(id: Int, x: Double, y: Double)

  case class Acenter(id: Int, x: Double, y: Double)

  class Test1(epsilon: Double) extends Serializable {

    def calculateDisksLeft(p1: Apoint, p2: Apoint, r2: Double): Acenter = {
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

      Acenter(0, h1, k1)
    }

    def calculateDisksRight(p1: Apoint, p2: Apoint, r2: Double): Acenter = {
      val X: Double = p1.x - p2.x
      val Y: Double = p1.y - p2.y
      val D2: Double = math.pow(X, 2) + math.pow(Y, 2)
      if (D2 == 0) throw new UnsupportedOperationException("Identical points...")
      val root: Double = math.sqrt(math.abs(4.0 * (r2 / D2) - 1.0))
      //val h1: Double = ((X + Y * root) / 2) + pair.getFloat(4)
      val h2: Double = ((X - Y * root) / 2) + p2.x
      //val k1: Double = ((Y - X * root) / 2) + pair.getFloat(5)
      val k2: Double = ((Y + X * root) / 2) + p2.y

      Acenter(0, h2, k2)
    }

    def run(pairsRDD: RDD[Row]): RDD[Acenter] = {
      val r2: Double = math.pow(epsilon / 2, 2)
      val centers = pairsRDD
        .filter((row: Row) => row.getInt(0) != row.getInt(3))
        .map { (row: Row) =>
          val p1 = Apoint(row.getInt(0), row.getDouble(1), row.getDouble(2))
          val p2 = Apoint(row.getInt(3), row.getDouble(4), row.getDouble(5))
          calculateDisksLeft(p1, p2, r2)
        }
      centers
    }
  }

  def main(args: Array[String]): Unit = {
    val master: String = "local[*]"
    var filename: String = "/opt/Datasets/Beijing/P10K.csv"
    val epsilon: Double = 10.0
    val mu: Int = 3
    val logs: String = "ERROR"

    val simba = SimbaSession
      .builder()
      .master(master)
      .appName("Benchmark")
      .config("simba.index.partitions", "256")
      .getOrCreate()
    simba.sparkContext.setLogLevel(logs)

    import simba.simbaImplicits._
    import simba.implicits._

    for (dataset <- 10 to 100 by 10; epsilon <- 10.0 to 200.0 by 10.0) {
      filename = s"/opt/Datasets/Beijing/P${dataset}K.csv"
      val tag = filename.substring(filename.lastIndexOf("/") + 1).split("\\.")(0).substring(1)
      val p1 = simba.sparkContext.textFile(filename)
        .map(_.split(","))
        .map(p => Apoint(p(0).trim.toInt, p(1).trim.toDouble, p(2).trim.toDouble))
        .toDF("id1", "x1", "y1")

      var time1 = System.currentTimeMillis()
      p1.index(RTreeType, "p1RT", Array("x1", "y1"))
      val p2 = p1.toDF("id2", "x2", "y2")

      val test1 = new Test1(epsilon)
      val pairsRDD = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon).rdd
      val centers = test1.run(pairsRDD).toDF()
      val n = centers.count()
      var time2 = System.currentTimeMillis()
      val time = (time2 - time1) / 1000.0
      println(s"PBFE,$epsilon,$tag,$n,$time")
    }
    //centers.index(RTreeType, "centeresRT", Array("x", "y"))
  }
}
