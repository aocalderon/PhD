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
        MaximalFinder.EPSILON = 100.0
        MaximalFinder.MU = 3
        val timestamps = d.map(datapoint => datapoint.t).distinct.sort("value").collect.toList

        println("\nTesting from files...")
        val g0 = simba.sparkContext.textFile(s"file:///home/acald013/PhD/Y3Q1/Datasets/s1.txt").map(_.split(",").toList.map(_.trim.toInt))
        g0.foreach(println)
        val g1 = simba.sparkContext.textFile(s"file:///home/acald013/PhD/Y3Q1/Datasets/s2.txt").map(_.split(",").toList.map(_.trim.toInt))
        g1.foreach(println)

        var time1: Long = System.currentTimeMillis()
        val g = g0.toDS.crossJoin(g1.toDS).show
        var time2: Long = System.currentTimeMillis()
        println("Cross Join in %d ms...".format(time2-time1))

        println("\nTesting from MaximalFinder...")
        var timestamp = timestamps.head
        var currentPoints = d.filter(datapoint => datapoint.t == timestamp).map(datapoint => MaximalFinder.SP_Point(datapoint.id, datapoint.x, datapoint.y))
        val f0: RDD[List[Int]] = MaximalFinder.run(currentPoints, timestamp, simba)
        //f0.persist
        f0.foreach(println)

        timestamp = timestamps(1)
        currentPoints = d.filter(datapoint => datapoint.t == timestamp).map(datapoint => MaximalFinder.SP_Point(datapoint.id, datapoint.x, datapoint.y))
        val f1: RDD[List[Int]] = MaximalFinder.run(currentPoints, timestamp, simba)
        //f1.persist
        f1.foreach(println)

        time1 = System.currentTimeMillis()
        val f = f0.repartition(2).toDS.crossJoin(f1.repartition(2).toDS).show
        time2 = System.currentTimeMillis()
        println("Cross Join in %d ms...".format(time2-time1))

        dataset.dropIndexByName("dRT")
        simba.close()
    }
}
