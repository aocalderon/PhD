package main

import org.apache.spark.sql.{SQLContext, Point, Row}
import org.apache.spark.{SparkConf, SparkContext}

object PBFE {
  case class PointItem(id: Int, x: Double, y: Double)
  case class Pair(id1: Int, id2: Int, x1: Double, y1: Double, x2: Double, y2: Double)
  
  var epsilon: Double = 1.0
  var r2 = math.pow(epsilon/2,2)
  var X = 0.0
  var Y = 0.0
  var D2 = 0.0
  var root = 0.0
  var h1 = 0.0
  var h2 = 0.0
  var k1 = 0.0
  var k2 = 0.0
  
  def calculateDisks(pair: Row) : Pair = {
    X = pair.getDouble(1) - pair.getDouble(4)
    Y = pair.getDouble(2) - pair.getDouble(5)
    D2 = math.pow(X, 2) + math.pow(Y, 2)
    if (D2 == 0)
        null
    root = math.pow(math.abs(4.0 * (r2 / D2) - 1.0), 0.5)
    h1 = ((X + Y * root) / 2) + pair.getDouble(4)
    h2 = ((X - Y * root) / 2) + pair.getDouble(4)
    k1 = ((Y - X * root) / 2) + pair.getDouble(5)
    k2 = ((Y + X * root) / 2) + pair.getDouble(5)
    
    Pair(pair.getInt(0), pair.getInt(3), h1, k1, h2, k2)
  }
  
  def main(args: Array[String]) : Unit = {
    val sparkConf = new SparkConf().setAppName("PBFE").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    if(args.length == 2)
      sc.setLogLevel("ERROR")
    else
      sc.setLogLevel(args(2))
    sqlContext.setConf("spark.sql.shuffle.partitions", 4.toString)
    sqlContext.setConf("spark.sql.sampleRate", 1.toString)
    sqlContext.setConf("spark.sql.partitioner.strTransferThreshold", 1000000.toString)

    import sqlContext.implicits._

    val filename = args(0)
    epsilon = args(1).toDouble
    val tag = filename.substring(filename.lastIndexOf("/") + 1).split("\\.")(0).substring(1)

    val p1 = sc.textFile(filename).map(_.split(",")).map(p => PointItem(p(0).trim.toInt, p(1).trim.toDouble, p(2).trim.toDouble)).toDF()
    val p2 = p1.toDF("id2", "x2", "y2")

    val time1 = System.currentTimeMillis()
    val pairs = p1.distanceJoin(p2, Point(p1("x"), p1("y")), Point(p2("x2"), p2("y2")), epsilon)
    val disks = pairs.rdd
       .filter( (x:Row) => x.getInt(0) > x.getInt(3) )
       .map( (x: Row) => calculateDisks(x) )
    var n = disks.count()
    // pairs.collect()
    val time2 = System.currentTimeMillis()
    println("PBFE," + epsilon + "," + tag + "," + 2 * n + "," + (time2 - time1) / 1000.0)

    sc.stop()
  }
}
