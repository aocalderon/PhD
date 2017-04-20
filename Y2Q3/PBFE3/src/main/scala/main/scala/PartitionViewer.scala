package main.scala

import edu.utah.cs.simba.SimbaContext
import edu.utah.cs.simba.index.RTreeType
import org.apache.spark.{SparkConf, SparkContext}
import org.osgeo.proj4j._

/**
  * Created by and on 3/20/17.
  */

object PartitionViewer {

  case class PointItem(id: Int, x: Double, y: Double)

  var master: String = "local[*]"
  var filename: String = "/opt/Datasets/Beijing/P10K.csv"
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

    import simbaContext.SimbaImplicits._
    import simbaContext.implicits._

    val points = sc.textFile(filename,10)
      .map(_.split(","))
      .map(p => PointItem(p(0).trim.toInt, p(1).trim.toDouble, p(2).trim.toDouble))
      .toDF()
    println(points.count())
    points index(RTreeType, "rt", Array("x", "y"))

    val mbrs = points.rdd.mapPartitionsWithIndex{ (index, iterator) =>
      val csName1 = "EPSG:4799"
      val csName2 = "EPSG:4326"
      val ctFactory = new CoordinateTransformFactory
      val csFactory = new CRSFactory
      /*
           * Create {@link CoordinateReferenceSystem} & CoordinateTransformation.
           * Normally this would be carried out once and reused for all transformations
           */
      val crs1 = csFactory.createFromName(csName1)
      val crs2 = csFactory.createFromName(csName2)
      val trans = ctFactory.createTransform(crs1, crs2)
      /*
           * Create input and output points.
           * These can be constructed once per thread and reused.
           */
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
      val p = new ProjCoordinate
      val pmin = new ProjCoordinate
      val pmax = new ProjCoordinate
      p.x = min_x
      p.y = min_y
      trans.transform(p, pmin)
      p.x = max_x
      p.y = max_y
      trans.transform(p, pmax)
      val wkt = "POLYGON ((" + pmin.x + " " + pmin.y + "," + pmax.x + " " + pmin.y + "," + pmax.x + " " + pmax.y + "," + pmin.x + " " + pmax.y + "," + pmin.x + " " + pmin.y + "))"
      List(s"$index;$min_x;$min_y;$max_x;$max_y;$wkt").iterator
    }

    mbrs.foreach(println)
    mbrs.saveAsTextFile("mbrs")
    sc.stop()
  }
}