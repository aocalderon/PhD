package main.scala

import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index.RTreeType
import org.apache.spark.sql.types.StructType

/**
  * Created by and on 5/8/17.
  */
object Demo {
  case class POI(pid: Long, poi_lon: Double, poi_lat: Double, tags: String)
  case class Car(cid:Long, car_lon: Double, car_lat: Double)

  def main(args: Array[String]): Unit = {
    val master: String = "local[*]"

    val simba = SimbaSession
      .builder()
      .master(master)
      .appName("Demo")
      .config("simba.index.partitions", "16")
      .getOrCreate()
    simba.sparkContext.setLogLevel("ERROR")

    import simba.implicits._
    import simba.simbaImplicits._

    var schema = ScalaReflection.schemaFor[POI].dataType.asInstanceOf[StructType]
    val poisDS = simba.read.
      option("header", "true").
      schema(schema).
      csv("samplePOIs.csv").
      as[POI]

    poisDS.show(5)
    poisDS.printSchema

    poisDS.index(RTreeType, "poisRT", Array("poi_lon", "poi_lat"))

    schema = ScalaReflection.schemaFor[Car].dataType.asInstanceOf[StructType]
    val carsDS = simba.read.
      option("header", "true").
      schema(schema).
      csv("sampleCars.csv").
      as[Car]

    carsDS.show(5)
    carsDS.printSchema

    carsDS.index(RTreeType, "carsRT", Array("car_lon", "car_lat"))

    val dj = poisDS
              .distanceJoin(carsDS, Array("poi_lon", "poi_lat"), Array("car_lon", "car_lat"), 200.0)
    dj.select("cid", "tags").show(20)
    println(dj.count())
  }
}
