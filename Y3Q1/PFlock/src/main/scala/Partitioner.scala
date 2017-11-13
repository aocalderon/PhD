import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.simba.index._
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType

/**
  * Created by and on 3/20/17.
  */

object Partitioner {

  case class SP_Point(id: Int, x: Double, y: Double)

  val master: String = "local[*]"
  val logs: String = "INFO"
  
  def toWKT(minx: Double, miny: Double, maxx: Double, maxy: Double): String = "POLYGON (( %f %f, %f %f, %f %f, %f %f, %f %f ))".
    format(
      minx, maxy,
      maxx, maxy,
      maxx, miny,
      minx, miny,
      minx, maxy
    )  

  def main(args: Array[String]): Unit = {
    val dataset: String = "B80K"
    val path: String = "Y3Q1/Datasets/"
    val extension: String = "csv"
    val partitions: String = "4"

    val simba = SimbaSession
      .builder()
      .master(master)
      .appName("Partitioner")
      .config("simba.index.partitions", partitions)
      .getOrCreate()

    import simba.implicits._
    import simba.simbaImplicits._
    simba.sparkContext.setLogLevel(logs)
    
    val POINT_SCHEMA = ScalaReflection.schemaFor[SP_Point].
        dataType.
        asInstanceOf[StructType]
            
    val phd_home = scala.util.Properties.
        envOrElse("PHD_HOME", "/home/and/Documents/PhD/Code/")
	val filename = s"$phd_home$path$dataset.$extension"
    val points = simba.
        read.option("header", "false").
        schema(POINT_SCHEMA).csv(filename).
        as[SP_Point]
    println(points.count())
    println(points.rdd.getNumPartitions)
    points.index(RTreeType, "rt", Array("x", "y"))
    println(points.rdd.getNumPartitions)
    
	val midx = 25241
	val midy1 = 21078
	val midy2 = 20834
    val delta = 50.5
    val data = points.rdd.mapPartitionsWithIndex{ (index, data) =>
      data.
      filter{ point => 
          (point.x < midx-delta && point.y > midy1+delta) || 
          (point.x < midx-delta && point.y < midy1-delta) || 
          (point.x > midx+delta && point.y > midy2+delta) || 
          (point.x > midx+delta && point.y < midy2-delta) 
      }.
      map(point => (index, point.id, point.x, point.y)).toIterator
    }
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
      List(toWKT(min_x,min_y,max_x,max_y), min_x,min_y,max_x,max_y, index, size).iterator
    }.
    foreach(println)	
    
	import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}
	var writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("/tmp/B20K_a.csv")))
    data.
      filter(point => point._1 == 0).
      map(point => "%d,%f,%f\n".format(point._2, point._3, point._4)).
      collect.toList.foreach(writer.write)
    writer.close()	

	writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("/tmp/B20K_b.csv")))
    data.
      filter(point => point._1 == 1).
      map(point => "%d,%f,%f\n".format(point._2, point._3, point._4)).
      collect.toList.foreach(writer.write)
    writer.close()	
	
	writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("/tmp/B20K_c.csv")))
    data.
      filter(point => point._1 == 2).
      map(point => "%d,%f,%f\n".format(point._2, point._3, point._4)).
      collect.toList.foreach(writer.write)
    writer.close()	

	writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("/tmp/B20K_d.csv")))
    data.
      filter(point => point._1 == 3).    
      map(point => "%d,%f,%f\n".format(point._2, point._3, point._4)).
      collect.toList.foreach(writer.write)
    writer.close()	

    simba.stop()
  }
}
