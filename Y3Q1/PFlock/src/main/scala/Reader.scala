import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.types.StructType
import org.rogach.scallop.{ScallopConf, ScallopOption}

object Reader {
  case class APoint(x: Double, y: Double, t: Int, id: Int)

  def main(args: Array[String]): Unit = {
    var log = List.empty[String]
    log = log :+ s"""{"content":"Starting app...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
    // Reading arguments from command line...
    val conf = new Conf(args)
    // Tuning master and number of cores...
    var MASTER = conf.master()
    if (conf.cores() == 1) {
      MASTER = "local[1]"
    }
    // Setting parameters...
    val POINT_SCHEMA = ScalaReflection.schemaFor[APoint].dataType.asInstanceOf[StructType]
    // Starting a session...
    log = log :+ s"""{"content":"Setting paramaters...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
    val simba = SimbaSession
      .builder()
      .master(MASTER)
      .appName("Reader")
      .config("simba.index.partitions", s"${conf.partitions()}")
      .config("spark.cores.max", s"${conf.cores()}")
      .getOrCreate()
    simba.sparkContext.setLogLevel(conf.logs())
    // Calling implicits...
    import simba.implicits._
    import simba.simbaImplicits._
    val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "Please, set up PHD_HOME environment variable...")
    val filename = s"${phd_home}${conf.path()}${conf.filename()}.${conf.extension()}"
    println(filename)
    val dataset = simba.read
      .option("header", "false")
      .schema(POINT_SCHEMA)
      .csv(filename)
      .as[APoint]
    dataset.show()
    val timestamps = dataset.map(point => point.t).distinct().collect().sorted
    timestamps.foreach{ timestamp =>
      val points = dataset
		.filter(datapoint => datapoint.t == timestamp)
		.map(datapoint => PFlock.APoint(datapoint.id, datapoint.x, datapoint.y))
      println("%d points in time %d".format(points.count(), timestamp))
      PFlock.run(points, timestamp, 10.0, 8, simba)
    }

    simba.close()
  }

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val partitions: ScallopOption[Int] = opt[Int](default = Some(32))
    val cores: ScallopOption[Int] = opt[Int](default = Some(4))
    val master: ScallopOption[String] = opt[String](default = Some("local[*]"))
    val logs: ScallopOption[String] = opt[String](default = Some("ERROR"))
    val output: ScallopOption[String] = opt[String](default = Some("output"))
    val phd_home: ScallopOption[String] = opt[String](default = sys.env.get("PHD_HOME"))
    val path: ScallopOption[String] = opt[String](default = Some("Y3Q1/Datasets/"))
    val filename: ScallopOption[String] = opt[String](default = Some("Berlin_N277K_A18K_T15"))
    val extension: ScallopOption[String] = opt[String](default = Some("csv"))

    verify()
  }

}
