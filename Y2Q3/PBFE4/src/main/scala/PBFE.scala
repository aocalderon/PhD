package main.scala

import java.util

import org.apache.spark.sql.simba.SimbaSession
import ca.pfv.spmf.algorithms.frequentpatterns.fpgrowth.AlgoFPMax
import org.apache.spark.sql.Row
import java.util.Calendar

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.simba.index.RTreeType

/**
  * Created by and on 3/20/17.
  */

object PBFE {

  case class PointItem(id: Int, x: Double, y: Double)

  case class Pair(id1: Int, id2: Int, x1: Double, y1: Double, x2: Double, y2: Double)

  //  var master: String = "local[*]"
  //  var epsilon: Double = 100.0
  //  var mu: Integer = 3
  //  var filename: String = "/opt/Datasets/Beijing/P10K.csv"
  //  var logs: String = "ERROR"

  var X = 0.0
  var Y = 0.0
  var D2 = 0.0
  var root = 0.0
  var h1 = 0.0
  var h2 = 0.0
  var k1 = 0.0
  var k2 = 0.0

  def calculateDisks(pair: Row, r2: Double): Pair = {
    X = pair.getDouble(1) - pair.getDouble(4)
    Y = pair.getDouble(2) - pair.getDouble(5)
    D2 = math.pow(X, 2) + math.pow(Y, 2)
    if (D2 == 0) throw new UnsupportedOperationException("Identical points...")
    root = math.pow(math.abs(4.0 * (r2 / D2) - 1.0), 0.5)
    h1 = ((X + Y * root) / 2) + pair.getDouble(4)
    h2 = ((X - Y * root) / 2) + pair.getDouble(4)
    k1 = ((Y - X * root) / 2) + pair.getDouble(5)
    k2 = ((Y + X * root) / 2) + pair.getDouble(5)

    Pair(pair.getInt(0), pair.getInt(3), h1, k1, h2, k2)
  }

  def main(args: Array[String]): Unit = {
    val master = args(0)
    val filename = args(1)
    val epsilon = args(2).toDouble
    val mu = args(3).toInt
    val logs = args(4)

    val r2: Double = math.pow(epsilon / 2, 2)
    val simbaSession = SimbaSession
      .builder()
      .master(master)
      .appName("PBFE")
      .config("simba.index.partitions", "128")
      .getOrCreate()

    import simbaSession.simbaImplicits._
    import simbaSession.implicits._
    import scala.collection.JavaConversions._


    val sc = simbaSession.sparkContext
    sc.setLogLevel(logs)

    val tag = filename.substring(filename.lastIndexOf("/") + 1).split("\\.")(0).substring(1)

    val p1 = sc.textFile(filename)
      .map(_.split(","))
      .map(p => PointItem(p(0).trim.toInt, p(1).trim.toDouble, p(2).trim.toDouble))
      .toDF
    p1.index(RTreeType, "pointsRT", Array("x", "y"))
    val p2 = p1.toDF("id2", "x2", "y2")

    p1.count()

    var time1 = System.currentTimeMillis()
    val pairs = p1.distanceJoin(p2, Array("x", "y"), Array("x2", "y2"), epsilon)
    val disks = pairs.rdd.filter((x: Row) => x.getInt(0) > x.getInt(3)).map((x: Row) => calculateDisks(x, r2))
    val ndisks = disks.count()
    var time2 = System.currentTimeMillis()
    val diskGenerationTime = (time2 - time1) / 1000.0

    //disks.map(())

    val centers1 = disks.toDF.select("x1", "y1")
    val centers2 = disks.toDF.select("x2", "y2")
    val centers = centers1.union(centers2)
    centers.index(RTreeType, "centersRT", Array("x1", "y1"))


    val membersRDD = centers.distanceJoin(p1, Array("x1", "y1"), Array("x", "y"), (epsilon / 2) + 0.01)
        .select("x1", "y1", "id")
        // TODO: run the group by here...
        .show()

        //.map { d => ( (d(0).asInstanceOf[Double], d(1).asInstanceOf[Double]) , d(2).asInstanceOf[Integer] ) }

/*    val members = membersRDD.groupByKey()
      .map{ m => ( m._1._1, m._1._2, m._2.toArray[Integer] ) }
      .toDF("x", "y", "IDs")
      .index(RTreeType, "membersRT", Array("x", "y"))

    val temp = members.rdd.mapPartitionsWithIndex{ (index, partition) =>
      System.out.println(s"$index : ")
      partition.foreach(println)
      partition.toIterator
      //val b = {
      //  partition.map { t => new util.ArrayList(t.map(_.asInstanceOf[Integer])) }.toBuffer
      //}
      //val ts = new util.ArrayList(b)
      //val fpmax = new AlgoFPMax
      //val itemsets = fpmax.runAlgorithm(ts, 1)
      //itemsets.getItemsets(mu).iterator()
    }

    temp.foreach(println)
    temp.count()*/

    /**************************************
      * Begin of tests...
      *************************************/



    /**************************************
      * End of tests...
      *************************************/

    /*
      val arrList = new ArrayList[Integer]()
      x.split(" ").map(y => arrList.add(y.toInt))
      Collections.sort(arrList)
      ts.add(arrList)
    */

    /*
    val minsup = 1
    time1 = System.currentTimeMillis()
    val dataset = new Dataset(ts)
    val lcm = new AlgoLCM
    var itemsets = lcm.runAlgorithm(minsup, dataset)
    //lcm.printStats
    //itemsets.printItemsets
    time2 = System.currentTimeMillis()
    val lcmTime = (time2 - time1) / 1000.0
    val lcmNItemsets = itemsets.countItemsets(3)
    */

/*    time1 = System.currentTimeMillis()
    val fpmax = new AlgoFPMax
    val itemsets = fpmax.runAlgorithm(new util.ArrayList(temp.toBuffer) , 1)
    //fpmax.printStats
    //itemsets.printItemsets
    time2 = System.currentTimeMillis()
    val fpMaxTime = (time2 - time1) / 1000.0
    val fpMaxNItemsets = itemsets.countItemsets(3)


    println("PBFE3,"
      + epsilon + ","
      + tag + ","
      + 2 * ndisks + ","
      + diskGenerationTime + ","
      + fpMaxTime + ","
      + fpMaxNItemsets + ","
      + Calendar.getInstance().getTime)*/

    sc.stop()
  }
}
