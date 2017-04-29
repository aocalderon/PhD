package main.scala

import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index._

/**
  * Created by and on 3/20/17.
  */

object PartitionViewer {

  case class PointItem(id: Int, x: Double, y: Double)

  var master: String = "local[*]"
  var filename: String = "/opt/Datasets/Beijing/P10K.csv"
  var epsilon: Double = 10.0
  var logs: String = "ERROR"

  def main(args: Array[String]): Unit = {
//    master = args(0)
//    filename = args(1)
//    logs = args(2)

    val simbaSession = SimbaSession
      .builder()
      .master(master)
      .appName("PartitionViewer")
      .config("simba.index.partitions", "256")
      .getOrCreate()

    import simbaSession.implicits._
    import simbaSession.simbaImplicits._

    val sc = simbaSession.sparkContext
    sc.setLogLevel(logs)

    val points = sc.textFile(filename,10)
      .map(_.split(","))
      .map(p => PointItem(id = p(0).trim.toInt, x = p(1).trim.toDouble, y = p(2).trim.toDouble))
      .toDF()
    println(points.count())
    points.index(RTreeType, "rt", Array("x", "y"))

    val mbrs = points.rdd.mapPartitionsWithIndex{ (index, iterator) =>
      var min_x: Double = Double.MaxValue
      var min_y: Double = Double.MaxValue
      var max_x: Double = Double.MinValue
      var max_y: Double = Double.MinValue

      var size: Int = 0

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
        size += 1
      }
      List((min_x,min_y,max_x,max_y, s"$index", size)).iterator
    }

    val gson = new GeoGSON("4799")
    mbrs.collect().foreach {row =>
      gson.makeMBR(row._1,row._2,row._3,row._4,row._5, row._6)
    }
    gson.saveGeoJSON("out/RTree_P100K.json")

    val gson2 = new GeoGSON("4799")
    mbrs.collect().foreach {row =>
      if(row._3 - row._1 > epsilon && row._4 - row._2 > epsilon){
        gson2.makeMBR(row._1 + epsilon,row._2 + epsilon,row._3 - epsilon,row._4 - epsilon,row._5, row._6)
      }
    }
    gson2.saveGeoJSON("out/RTree_P100K_buffer.json")

    mbrs.map(r => r._6).toDF("n").agg(Map("n" -> "avg")).show()

    sc.stop()
  }
}