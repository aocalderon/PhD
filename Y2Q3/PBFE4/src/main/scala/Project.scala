package main.scala

import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index.RTreeType
import org.apache.spark.sql.types.StructType

import scala.collection.JavaConversions._

/**
  * Created by and on 5/11/17.
  */
object Project {
  case class POI(pid: Long, poi_lon: Double, poi_lat: Double, tags: String)
  case class Trajectory(tid: Long, oid: Long, lon: Double, lat: Double, time: String)
  case class Grid(glon: Double, glat: Double)

  def main(args: Array[String]): Unit = {
    val master: String = "local[*]"

    val simba = SimbaSession
      .builder()
      .master(master)
      .appName("Project")
      .config("simba.index.partitions", "64")
      .getOrCreate()
    simba.sparkContext.setLogLevel("ERROR")

    import simba.implicits._
    import simba.simbaImplicits._

    val schema = ScalaReflection.schemaFor[Trajectory].dataType.asInstanceOf[StructType]
    val trajectories = simba.read.
      option("header", "false").
      schema(schema).
      //csv("out/B40Trajs.csv").
      csv("/opt/GISData/Geolife_Trajectories_1.3/beijing2.csv").
      as[Trajectory]

    trajectories.show(25)
    trajectories.printSchema

    val cx = -323750.0
    val cy = 4471800.0
    val extend = 100000.0
    var clip = trajectories.range(Array("lon", "lat"),
                                  Array(cx - extend, cy - extend),
                                  Array(cx + extend, cy + extend)).toDF().as[Trajectory]
    println(clip.count())
    clip = clip.distinct()
    println(clip.count())

    clip.index(RTreeType, "clipRT", Array("lon", "lat"))

    val mbrs = clip.rdd.mapPartitionsWithIndex{ (index, iterator) =>
      var min_x: Double = Double.MaxValue
      var min_y: Double = Double.MaxValue
      var max_x: Double = Double.MinValue
      var max_y: Double = Double.MinValue

      var size: Int = 0

      iterator.foreach{ row =>
        val x = row.lon
        val y = row.lat
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
    gson.saveGeoJSON("out/clip_RTree.json")

    val minx = cx - extend
    val miny = cy - extend
    val maxx = cx + extend
    val maxy = cy + extend

    val x = simba.sparkContext.parallelize(minx to maxx by 10000)
    val y = simba.sparkContext.parallelize(miny to maxy by 10000)
    val grid = x.cartesian(y).map(cell => Grid(cell._1, cell._2)).toDS()
    grid.index(RTreeType, "gridRT", Array("glon", "glat"))
    val results = trajectories.knnJoin(grid, Array("lon", "lat"), Array("glon", "glat"), 1).
                                select("tid", "oid", "glon", "glat").
                                groupBy("glon", "glat").
                                count().
                                collect()
    //results.foreach(print)
    import au.com.bytecode.opencsv.CSVWriter
    import java.io.BufferedWriter
    import java.io.FileWriter
    val out = new BufferedWriter(new FileWriter("results.csv"))
    val writer = new CSVWriter(out)
    val r = results.map(row => Array(row.getDouble(0).toString,
                                      row.getDouble(1).toString,
                                      row.getLong(2).toString)).toList
    r.foreach(println)
    writer.writeAll(r)
    out.close()
  }
}
