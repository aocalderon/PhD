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
  case class POI(pid: Long, tags: String, poi_lon: Double, poi_lat: Double)
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

    val schema = ScalaReflection.schemaFor[POI].dataType.asInstanceOf[StructType]
    var pois = simba.read.
      option("header", "true").
      schema(schema).
      //csv("out/B40Trajs.csv").
      csv("/opt/GISData/POIs.csv").
      as[POI]

    pois.filter("poi_lon IS NULL").collect().foreach(println)

    println(pois.count())
    pois.printSchema

    val cx = -323750.0
    val cy = 4471800.0
    val extend = 10000.0
    val minx = cx - extend
    val miny = cy - extend
    val maxx = cx + extend
    val maxy = cy + extend

//    val clip = trajectories.range(Array("lon", "lat"),
//                                  Array(minx, miny),
//                                  Array(maxx, maxy)).
//                            toDF().as[Trajectory]
//    val clip = trajectories.range(Array("poi_lon", "poi_lat"),
//      Array(-332729.310,4456050.000),
//      Array(-316725.862,4469518.966)).
//      toDF().as[POI]
//
//    println(clip.count())
//    clip.write.csv("/tmp/clip")
//    clip.index(RTreeType, "clipRT", Array("lon", "lat"))
//
    val x = simba.sparkContext.parallelize(minx to maxx by 1000)
    val y = simba.sparkContext.parallelize(miny to maxy by 1000)
    val grid = x.cartesian(y).map(cell => Grid(cell._1, cell._2)).toDS()
//    grid.index(RTreeType, "gridRT", Array("glon", "glat"))
//    val results = clip.knnJoin(grid, Array("lon", "lat"), Array("glon", "glat"), 1).
//                                select("tid", "oid", "glon", "glat").
//                                groupBy("glon", "glat").
//                                count().
//                                collect()
//    //results.foreach(print)
//    import au.com.bytecode.opencsv.CSVWriter
//    import java.io.BufferedWriter
//    import java.io.FileWriter
//    val out = new BufferedWriter(new FileWriter("results.csv"))
//    val writer = new CSVWriter(out)
//    val r = results.map(row => Array(row.getDouble(0).toString,
//                                      row.getDouble(1).toString,
//                                      row.getLong(2).toString)).toList
//    r.foreach(println)
//    writer.writeAll(r)
//    out.close()
  }
}
