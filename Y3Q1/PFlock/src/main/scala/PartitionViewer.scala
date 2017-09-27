import Misc.GeoGSON
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index._

/**
  * Created by and on 3/20/17.
  */

object PartitionViewer {

  case class PointItem(id: Int, x: Double, y: Double)

  val master: String = "local[*]"
  val logs: String = "ERROR"

  def main(args: Array[String]): Unit = {
    val EPSG: String = "3068"
    val dataset: String = "Berlin"
    val filename: String = "/home/and/Documents/PhD/Code/Y2Q4/BerlinSample/B160K_3068.csv"
    val path: String = "output/"
    val epsilon: Double = 10.0
    val partitions: String = "4096"
//    master = args(0)
//    filename = args(1)
//    logs = args(2)

    val simbaSession = SimbaSession
      .builder()
      .master(master)
      .appName("PartitionViewer")
      .config("simba.index.partitions", partitions)
      .getOrCreate()

    import simbaSession.implicits._
    import simbaSession.simbaImplicits._

    val sc = simbaSession.sparkContext
    sc.setLogLevel(logs)

    val points = sc.textFile(filename,10)
      .map(_.split(","))
      .map(p => PointItem(id = p(0).trim.toInt, x = p(1).trim.toDouble, y = p(2).trim.toDouble))
      .toDS()
    println(points.count())
    points.index(RTreeType, "rt", Array("x", "y"))


    val mbrs = points.rdd.mapPartitionsWithIndex{ (index, iterator) =>
      var min_x: Double = Double.MaxValue
      var min_y: Double = Double.MaxValue
      var max_x: Double = Double.MinValue
      var max_y: Double = Double.MinValue

      var size: Int = 0

      iterator.toList.foreach{row =>
        val x = row.x
        val y = row.y
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

    val gson = new GeoGSON(EPSG)
    mbrs.collect().foreach {row =>
      gson.makeMBR(row._1,row._2,row._3,row._4,row._5, row._6)
    }
    gson.saveGeoJSON(path + "RTree_" + dataset + "_" + EPSG + "_P" + partitions + ".geojson")

    val gson2 = new GeoGSON(EPSG)
    mbrs.collect().foreach {row =>
      if(row._3 - row._1 > epsilon && row._4 - row._2 > epsilon){
        gson2.makeMBR(row._1 + epsilon,row._2 + epsilon,row._3 - epsilon,row._4 - epsilon,row._5, row._6)
      }
    }
    gson2.saveGeoJSON(path + "RTree_" + dataset + "_" + EPSG + "_P" + partitions + "_E" + epsilon + ".geojson")

    mbrs.map(r => r._6).toDF("n").agg(Map("n" -> "avg")).show()

    sc.stop()
  }
}