package main

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SQLContext, Row}
import org.apache.spark.sql.Point

object DistanceJoin {
  case class PointItem(id: Int, x: Double, y: Double)
  case class Pair(id1: Int, id2: Int, x1: Double, y1: Double, x2: Double, y2: Double)

  def main(args: Array[String]) : Unit = {
    val sparkConf = new SparkConf().setAppName("DistanceJoin").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    val filename = "P10K.csv"
    val epsilon = 100

    val p1 = sc.textFile(filename).map(_.split(",")).map(p => PointItem(p(0).trim.toInt, p(1).trim.toDouble, p(2).trim.toDouble)).toDF()
    val p2 = p1.toDF("id2", "x2", "y2")

    val pairs = p1.distanceJoin(p2, Point(p1("x"), p1("y")), Point(p2("x2"), p2("y2")), epsilon)
    val disks = pairs.rdd
      .filter( (x:Row) => x.getInt(0) > x.getInt(3) )
      .map( (x: Row) => calculateDisks(x) )
    println(disks.count())

    sc.stop()
  }
}
