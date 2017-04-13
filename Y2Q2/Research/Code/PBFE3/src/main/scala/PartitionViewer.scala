package main.scala

import edu.utah.cs.simba.SimbaContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by and on 3/20/17.
  */

object PartitionViewer {

  case class PointItem(id: Int, x: Double, y: Double)

  var master: String = "local[*]"
  var filename: String = "/opt/Datasets/Beijing/P10K.csv"
  var logs: String = "INFO"

  def main(args: Array[String]): Unit = {
//    master = args(0)
//    filename = args(1)
//    logs = args(2)

    val sparkConf = new SparkConf()
      .setAppName("PartitionViewer")
      .setMaster(master)
    val sc = new SparkContext(sparkConf)
    sc.setLogLevel(logs)
    val simbaContext = new SimbaContext(sc)

    import simbaContext.implicits._
    import simbaContext.SimbaImplicits._

    val points = sc.textFile(filename,10)
      .map(_.split(","))
      .map(p => PointItem(p(0).trim.toInt, p(1).trim.toDouble, p(2).trim.toDouble))
    println(points.count())
    println(points.getNumPartitions)

    var time1 = System.currentTimeMillis()
    val temp = points.mapPartitionsWithIndex{ (index, iterator) =>
      iterator.toList.map(point => index + "," + point.x + "," + point.y).iterator
    }
    var time2 = System.currentTimeMillis()
    val partitionTime = (time2 - time1) / 1000.0
    temp.saveAsTextFile("output")

    sc.stop()
  }
}