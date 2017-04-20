package main.scala

import java.util
import java.util.Calendar

import ca.pfv.spmf.algorithms.frequentpatterns.fpgrowth.AlgoFPMax
import edu.utah.cs.simba.SimbaContext
import org.apache.spark.sql.Row
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by and on 3/20/17.
  */

object PBFE3 {

  case class PointItem(id: Int, x: Double, y: Double)

  case class Pair(id1: Int, id2: Int, x1: Double, y1: Double, x2: Double, y2: Double)

  var master: String = "local[*]"
  var epsilon: Double = 100.0
  var mu: Integer = 3
  var filename: String = "/opt/Datasets/Beijing/B89.csv"
  var logs: String = "ERROR"
  var r2: Double = math.pow(epsilon / 2, 2)
  val NCORES = 10

  var X = 0.0
  var Y = 0.0
  var D2 = 0.0
  var root = 0.0
  var h1 = 0.0
  var h2 = 0.0
  var k1 = 0.0
  var k2 = 0.0

  def calculateDisks(pair: Row): Pair = {
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
    master = args(0)
    filename = args(1)
    epsilon = args(2).toDouble
    mu = args(3).toInt
    logs = args(4)

    val sparkConf = new SparkConf()
      .setAppName("PBFE3")
      .setMaster(master)
    val sc = new SparkContext(sparkConf)
    sc.setLogLevel(logs)
    val simbaContext = new SimbaContext(sc)

    import simbaContext.SimbaImplicits._
    import simbaContext.implicits._

    import scala.collection.JavaConversions._

    val tag = filename.substring(filename.lastIndexOf("/") + 1).split("\\.")(0).substring(1)

    val p1 = sc.textFile(filename).map(_.split(",")).map(p => PointItem(p(0).trim.toInt, p(1).trim.toDouble, p(2).trim.toDouble)).toDF
    val p2 = p1.toDF("id2", "x2", "y2")
    p1.count()

    var time1 = System.currentTimeMillis()
    val pairs = p1.distanceJoin(p2, Array("x", "y"), Array("x2", "y2"), epsilon)
    val disks = pairs.rdd.filter((x: Row) => x.getInt(0) > x.getInt(3)).map((x: Row) => calculateDisks(x))
    val ndisks = disks.count()
    var time2 = System.currentTimeMillis()
    val diskGenerationTime = (time2 - time1) / 1000.0

    val centers1 = disks.toDF.select("x1", "y1")
    val centers2 = disks.toDF.select("x2", "y2")
    val centers = centers1.unionAll(centers2)

    val members = centers.distanceJoin(p1, Array("x1", "y1"), Array("x", "y"), (epsilon / 2) + 0.01)
      .select("x1", "y1", "id")
      .rdd
      .map { d => (d(0) + ";" + d(1), d(2)) }
      .groupByKey()
      .map { m =>
        m._2.toBuffer
      }

    val temp = members.mapPartitions{ partition =>
      val b = partition.map{ t => new util.ArrayList(t.map(_.asInstanceOf[Integer])) }.toBuffer
      val ts = new util.ArrayList(b)

      val fpmax = new AlgoFPMax
      val itemsets = fpmax.runAlgorithm(ts, 1)
      itemsets.getItemsets(mu).iterator()
    }.collect()

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

    time1 = System.currentTimeMillis()
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
      + Calendar.getInstance().getTime)

    sc.stop()
  }
}
