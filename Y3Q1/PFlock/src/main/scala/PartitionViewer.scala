import Misc.GeoGSON
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index._
import org.slf4j.{Logger, LoggerFactory}

/**
  * Created by and on 3/20/17.
  */

object PartitionViewer {
  private val logger: Logger = LoggerFactory.getLogger("myLogger")
  case class PointItem(id: Int, x: Double, y: Double)

  def main(args: Array[String]): Unit = {
    val EPSG: String = "3068"
    val phd_home = "/home/acald013/PhD/"
    val path: String = "Y3Q1/Datasets/"
    val extension: String = ".csv"
    val dataset: String = args(0)
    val epsilon: Double = args(1)
    val partitions: String = args(2)
    val master: String = args(3)

    logger.info("Starting session...")
    val simbaSession = SimbaSession
      .builder()
      .master(master)
      .appName("PartitionViewer")
      .config("simba.index.partitions", partitions)
      .getOrCreate()
    import simbaSession.implicits._
    import simbaSession.simbaImplicits._
    val filename: String = "%s%s%s.%s".format(phd_home, path, dataset, extension)
    logger.info("Reading %s...".format(filename))
    val points = sc.textFile(filename,10)
      .map(_.split(","))
      .map(p => PointItem(id = p(0).trim.toInt, x = p(1).trim.toDouble, y = p(2).trim.toDouble))
      .toDS()
    logger.info("%d have been readed...".format(points.count()))
    logger.info("Indexing...")
    points.index(RTreeType, "pRT", Array("x", "y"))


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
