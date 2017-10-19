import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.simba.index.RTreeType
import org.apache.spark.rdd.DoubleRDDFunctions
import org.slf4j.Logger
import org.slf4j.LoggerFactory

object Test {
	val logger: Logger = LoggerFactory.getLogger("myLogger")
    case class SP_Point(id: Int, x: Double, y: Double)
    
    def main(args: Array[String]): Unit = {
        val POINT_SCHEMA = ScalaReflection.schemaFor[SP_Point].dataType.asInstanceOf[StructType] 
        val simba = SimbaSession.builder().
			master(args(3)).
			appName("Test").
			config("simba.rtree.maxEntriesPerNode", args(1)).
			config("simba.index.partitions", args(2)).
			config("spark.cores.max", args(4)).
			getOrCreate()
        import simba.implicits._
        import simba.simbaImplicits._
        val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
        val path = "Y3Q1/Datasets/"
        val dataset = args(0)
        val extension = "csv"
        val filename = "%s%s%s.%s".format(phd_home, path, dataset, extension)
        val points = simba.read.option("header", "false").schema(POINT_SCHEMA).csv(filename).as[SP_Point]
        val n = points.count()
        val time1 = System.currentTimeMillis()
        points.index(RTreeType, "pRT", Array("x", "y"))
        val time2 = System.currentTimeMillis()
        val time = (time2 - time1) / 1000.0
		val partition_sizes = points.rdd.mapPartitions{ it =>
		  List(it.size.toDouble).iterator
		}
		val partitions = points.rdd.getNumPartitions
		val sizes = new DoubleRDDFunctions(partition_sizes)
		val avg = sizes.mean()
		val sd = sizes.stdev()
		val variance = sizes.variance()
		val max = partition_sizes.max()
		val min = partition_sizes.min()
		logger.info("%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f".format(dataset, n, partitions, time, avg, sd, variance, min, max))
        points.dropIndexByName("pRT")
        simba.close()
    }
}
