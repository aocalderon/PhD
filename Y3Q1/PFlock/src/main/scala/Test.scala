import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.types.StructType
import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.apache.spark.sql.simba.index.RTreeType

object Test {
    case class ST_Point(x: Double, y: Double, t: Int, id: Int)
	def main(args: Array[String]): Unit = {
        val POINT_SCHEMA = ScalaReflection.schemaFor[ST_Point].dataType.asInstanceOf[StructType]
        val simba = SimbaSession.builder().master("local[7]").appName("Runner").config("simba.index.partitions", "64").config("spark.cores.max", "7").getOrCreate()
        import simba.implicits._
        import simba.simbaImplicits._
        val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
        val filename = s"${phd_home}Y3Q1/Datasets/Berlin_N15K_A1K_T15.csv"
        val dataset = simba.read.option("header", "false").schema(POINT_SCHEMA).csv(filename).as[ST_Point].filter(datapoint => datapoint.t < 120)
        val d = dataset.index(RTreeType, "dRT", Array("x", "y"))
        val c = d.count()
        d.cache
        PFlock.EPSILON = 100.0
        PFlock.MU = 3
        val timestamps = d.map(datapoint => datapoint.t).distinct.sort("value").collect.toList

        var timestamp = timestamps.head
        var currentPoints = d.filter(datapoint => datapoint.t == timestamp).map(datapoint => PFlock.SP_Point(datapoint.id, datapoint.x, datapoint.y))
        val f0: RDD[List[Int]] = PFlock.run(currentPoints, timestamp, simba)
        f0.foreach(println)

        timestamp = timestamps(1)
        currentPoints = d.filter(datapoint => datapoint.t == timestamp).map(datapoint => PFlock.SP_Point(datapoint.id, datapoint.x, datapoint.y))
        val f1: RDD[List[Int]] = PFlock.run(currentPoints, timestamp, simba)
        f1.foreach(println)

        val f = f0.cartesian(f1)
        f.foreach(println)
        println(f.count())     
        
        val g0 = simba.sparkContext.textFile(s"file://${phd_home}Y3Q1/Datasets/s1.txt").map(_.split(",").toList.map(_.trim.toInt))
        g0.foreach(println)
        val g1 = simba.sparkContext.textFile(s"file://${phd_home}Y3Q1/Datasets/s2.txt").map(_.split(",").toList.map(_.trim.toInt))
        g1.foreach(println)

        val g = g0.cartesian(g1)
        g.foreach(println)
        println(g.count())
        
        dataset.dropIndexByName("dRT")
        simba.close()

	}
}
