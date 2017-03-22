package main

import java.util

import org.apache.spark.sql.{Point, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import java.util.Calendar

import scala.collection.mutable.ListBuffer
import ca.pfv.spmf.algorithms.frequentpatterns.lcm.{AlgoLCM, Dataset}
import ca.pfv.spmf.patterns.itemset_array_integers_with_count.Itemset

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
    val sparkConf = new SparkConf().setAppName("PBFE")
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    if(args.length == 2)
      sc.setLogLevel("ERROR")
    else
      sc.setLogLevel(args(2))

    import sqlContext.implicits._

    val filename = args(0)
    epsilon = args(1).toDouble
    val tag = filename.substring(filename.lastIndexOf("/") + 1).split("\\.")(0).substring(1)

    val p1 = sc.textFile(filename).map(_.split(",")).map(p => PointItem(p(0).trim.toInt, p(1).trim.toDouble, p(2).trim.toDouble)).toDF()
    val p2 = p1.toDF("id2", "x2", "y2")


    var time1 = System.currentTimeMillis()
    val pairs = p1.distanceJoin(p2, Point(p1("x"), p1("y")), Point(p2("x2"), p2("y2")), epsilon)
    val disks = pairs.rdd.filter( (x:Row) => x.getInt(0) > x.getInt(3) ).map( (x: Row) => calculateDisks(x) )
    var n = disks.count()
    val time2 = System.currentTimeMillis()

    println("PBFE2," + epsilon + "," + tag + "," + 2*n + "," + (time2 - time1) / 1000.0 + "," + Calendar.getInstance().getTime())

    time1 = System.currentTimeMillis()
    var centers1 = disks.toDF().select("x1", "y1")
    var centers2 = disks.toDF().select("x2", "y2")
    var centers = centers1.unionAll(centers2)

    var members = centers.distanceJoin(p1, Point(centers("x1"), centers("y1")), Point(p1("x"), p1("y")), (epsilon/2) + 0.01)
      .select("x1", "y1", "id")
      .rdd
      .map{ d => (d(0) + "-" + d(1), d(2)) }
      .groupByKey()
      .map{ m => m._2.mkString(" ") }

    members.toDF().write.format("com.databricks.spark.csv").save("tdisks")
/*
    var nnn = members.count()
    println(nnn)

    val ts2 = new util.ArrayList[util.ArrayList[Integer]]()
    members.collect().foreach{ x =>
      val arrList = new util.ArrayList[Integer]()
      x.split(" ").map( y => arrList.add(y.toInt) )
      ts2.add(arrList)
    }
    val it = ts2.iterator()
    while(it.hasNext){
      println(it.next())
    }

    val minsup = 1 / nnn
    val t1 = new util.ArrayList[Integer]()
    t1.add(1)
    t1.add(Integer.valueOf(2))
    t1.add(Integer.valueOf(5))
    t1.add(Integer.valueOf(7))
    t1.add(Integer.valueOf(9))
    val t2 = new util.ArrayList[Integer]()
    t2.add(Integer.valueOf(1))
    t2.add(Integer.valueOf(3))
    t2.add(Integer.valueOf(5))
    t2.add(Integer.valueOf(7))
    t2.add(Integer.valueOf(9))
    val t3 = new util.ArrayList[Integer]()
    t3.add(Integer.valueOf(1))
    t3.add(Integer.valueOf(4))
    t3.add(Integer.valueOf(1))
    t3.add(Integer.valueOf(3))
    val t4 = new util.ArrayList[Integer]()
    t4.add(Integer.valueOf(1))
    t4.add(Integer.valueOf(3))
    t4.add(Integer.valueOf(4))
    t4.add(Integer.valueOf(5))
    t4.add(Integer.valueOf(6))
    val t5 = new util.ArrayList[Integer]()
    t5.add(Integer.valueOf(1))
    t5.add(Integer.valueOf(2))
    val t6 = new util.ArrayList[Integer]()
    t6.add(Integer.valueOf(2))
    t6.add(Integer.valueOf(1))
    val t7 = new util.ArrayList[Integer]()
    t7.add(Integer.valueOf(1))
    t7.add(Integer.valueOf(7))
    t7.add(Integer.valueOf(2))
    t7.add(Integer.valueOf(3))
    t7.add(Integer.valueOf(4))
    t7.add(Integer.valueOf(5))
    t7.add(Integer.valueOf(8))
    t7.add(Integer.valueOf(9))
    val t8 = new util.ArrayList[Integer]()
    t8.add(Integer.valueOf(6))
    t8.add(Integer.valueOf(1))
    t8.add(Integer.valueOf(2))
    val t9 = new util.ArrayList[Integer]()
    t9.add(Integer.valueOf(4))
    t9.add(Integer.valueOf(5))
    t9.add(Integer.valueOf(6))
    val t10 = new util.ArrayList[Integer]()
    t10.add(Integer.valueOf(8))
    t10.add(Integer.valueOf(2))
    t10.add(Integer.valueOf(5))
    val t11 = new util.ArrayList[Integer]()
    t11.add(Integer.valueOf(9))
    t11.add(Integer.valueOf(2))
    t11.add(Integer.valueOf(1))
    val t12 = new util.ArrayList[Integer]()
    t12.add(Integer.valueOf(1))
    t12.add(Integer.valueOf(2))
    t12.add(Integer.valueOf(4))
    t12.add(Integer.valueOf(8))
    t12.add(Integer.valueOf(9))
    val ts = new util.ArrayList[util.ArrayList[Integer]]
    ts.add(t1)
    ts.add(t2)
    ts.add(t3)
    ts.add(t4)
    ts.add(t5)
    ts.add(t6)
    ts.add(t7)
    ts.add(t8)
    ts.add(t9)
    ts.add(t10)
    ts.add(t11)
    ts.add(t12)
    val dataset = new Dataset(ts2)
    val algo = new AlgoLCM
    val itemsets = algo.runAlgorithm(minsup, dataset, null.asInstanceOf[String])
    algo.printStats()
    itemsets.printItemsets(dataset.getTransactions.size)
*/
    sc.stop()
  }
}
