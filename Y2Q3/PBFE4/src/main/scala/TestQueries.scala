package main.scala

import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index.RTreeType
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.simba.{DataFrame => SimbaDataFrame, Dataset => SimbaDataSet}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import java.sql.Timestamp

import org.apache.spark.sql.functions._

case class POI(pid: Long, tags: String, poi_lon: Double, poi_lat: Double)
case class Trajectory(cid: Int, eid: Int, lon: Double, lat: Double, dt: Timestamp)

/**
  * Created by Bradd
  */
object TestQueries {
  def main(args: Array[String]): Unit = {
    // Instantiate SimbaSession
    val simba = SimbaSession.
      builder().
      master("local[*]").
      appName("Demo").
      config("simba.index.partitions", "8").
      getOrCreate()
    simba.sparkContext.setLogLevel("ERROR")

    // Run Test
    RangeQueryA(simba)
    //RadiusQueryB(simba,5000)
    //GridQueryC(simba,500)
    //HotSpotQueryD(simba,100)
    // Close Simba
    //simba.close()
  }

  /** ********************************************
    * Query 1
    * ********************************************
    * Retrieve all the shops located inside the 3rd road ring of the city.
    * The coordinates which you should use are
    * (-332729.310,4456050.000) and (-316725.862,4469518.966)
    *
    * @param simba
    * *********************************************/
  def RangeQueryA(simba: SimbaSession): Unit = {
    import simba.implicits._
    import simba.simbaImplicits._

    // Read in necessary files
    val tableName = "p"
    val fileName = "/opt/GISData/POIs.csv"
    val schema = ScalaReflection.schemaFor[POI].dataType.asInstanceOf[StructType]
    val ds = simba.read.
      schema(schema).
      csv(fileName).
      as[POI]
    ds.createOrReplaceTempView(tableName)

    // Index and Filter Table
    val sql = "SELECT * FROM " + tableName + " WHERE tags LIKE '%shop=%'"
    val result = simba.sql(sql).as[POI]
    println("# of results: " + result.count())

    // Perform Range Query
    val bottomLeft = Array(-332729.310, 4456050.000)
    val topRight = Array(-316725.862, 4469518.966)
    val range = result.range(Array("poi_lon", "poi_lat"), bottomLeft, topRight).as[POI]
    println("# of results in range: " + range.count())
  }
}