package main.scala

import edu.utah.cs.simba.SimbaContext
import org.apache.spark.{SparkConf, SparkContext}
import edu.utah.cs.simba.index.RTreeType

/**
  * Created by and on 3/20/17.
  */

object PartitionViewer {

  case class PointItem(id: Int, x: Double, y: Double)

  var master: String = "local[*]"
  var filename: String = "/opt/Datasets/Beijing/B89.csv"
  var logs: String = "ERROR"

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
      .toDF()
    println(points.count())
    points.index(RTreeType, "rt", Array("x", "y"))

    var time1 = System.currentTimeMillis()
    val temp = points.rdd
      .mapPartitionsWithIndex{ (index, iterator) =>
        iterator.toList.map(point => index + "," + point(1) + "," + point(2)).iterator
      }
    var time2 = System.currentTimeMillis()
    val partitionTime = (time2 - time1) / 1000.0
    temp.saveAsTextFile("datasets")
    temp.foreach(println)
    val mbrs = points.rdd.mapPartitionsWithIndex{ (index, iterator) =>
      var min_x: Double = Double.MaxValue
      var min_y: Double = Double.MaxValue
      var max_x: Double = Double.MinValue
      var max_y: Double = Double.MinValue

      iterator.toList.foreach{row =>
        val x = row.getDouble(1)
        val y = row.getDouble(2)
        if(x < min_x){
          min_x = x
        }
        if(y < min_y){
          min_y = y
        }
        if(x > max_x){
          max_x = x
        }
        if(y > max_y){
          max_y = y
        }
      }
      List(index + "," + min_x + "," + min_y + "," + max_x + "," + max_y).iterator
    }
    mbrs.foreach(println)
    mbrs.saveAsTextFile("mbrs")
    sc.stop()
  }
}