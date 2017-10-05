import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.types.StructType
import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.slf4j.Logger
import org.slf4j.LoggerFactory

object Runner {
  private val LAYER_0 = 117
  private val LAYERS_N = 10

  case class ST_Point(x: Double, y: Double, t: Int, id: Int)

  def main(args: Array[String]): Unit = {
    // Setting a custom logger...
    val log: Logger = LoggerFactory.getLogger("myLogger")
    log.info(s"""{"content":"Starting app...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n""")
    // Reading arguments from command line...
    val conf = new Conf(args)
    // Tuning master and number of cores...
    var MASTER = conf.master()
    if (conf.cores() == 1) {
      MASTER = "local[1]"
    }
    // Setting parameters...
    val POINT_SCHEMA = ScalaReflection.schemaFor[ST_Point].dataType.asInstanceOf[StructType]
    // Starting a session...
    log.info(s"""{"content":"Setting paramaters...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n""")
    val simba = SimbaSession.builder().master(MASTER).appName("Runner").
		config("simba.index.partitions", s"${conf.partitions()}").
		config("spark.cores.max", s"${conf.cores()}").
		getOrCreate()
    simba.sparkContext.setLogLevel(conf.logs())
    // Calling implicits...
    import simba.implicits._
    import simba.simbaImplicits._
    val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
    val filename = s"$phd_home${conf.path()}${conf.dataset()}.${conf.extension()}"
    log.info("Reading %s ...".format(filename))
    val dataset = simba.read
      .option("header", "false")
      .schema(POINT_SCHEMA)
      .csv(filename)
      .as[ST_Point]
      .filter(datapoint => datapoint.t < LAYER_0 + LAYERS_N)
    dataset.cache()
    log.info("Number of points in dataset: %d".format(dataset.count()))
    val timestamps = dataset.map(datapoint => datapoint.t).distinct.sort("value").collect.toList
    // Setting PFlock...
    PFlock.EPSILON = conf.epsilon()
    PFlock.MU = conf.mu()
    PFlock.DATASET = conf.dataset()
    PFlock.CORES = conf.cores()
    PFlock.PARTITIONS = conf.partitions()
    // Running PFlock...
    var timestamp = timestamps.head
    var currentPoints = dataset
      .filter(datapoint => datapoint.t == timestamp)
      .map(datapoint => PFlock.SP_Point(datapoint.id, datapoint.x, datapoint.y))
    log.info("%d points in time %d".format(currentPoints.count(), timestamp))
    val f0: RDD[List[Int]] = PFlock.run(currentPoints, timestamp, simba)
    f0.cache()

    timestamp = timestamps(1)
    currentPoints = dataset
      .filter(datapoint => datapoint.t == timestamp)
      .map(datapoint => PFlock.SP_Point(datapoint.id, datapoint.x, datapoint.y))
    log.info("%d points in time %d".format(currentPoints.count(), timestamp))
    val f1: RDD[List[Int]] = PFlock.run(currentPoints, timestamp, simba)
    f1.cache()

    log.info("F0: " + f0.count())
    f0.foreach(println)
    log.info("F1: " + f1.count())
    f1.foreach(println)

    val g0 = simba.sparkContext
      .textFile(s"file://$phd_home${conf path()}s1.txt")
      .map(_.split(",").toList.map(_.trim.toInt))
    log.info("G0: " + g0.count())
    f0.foreach(println)
    val g1 = simba.sparkContext
      .textFile(s"file://$phd_home${conf.path()}s1.txt")
      .map(_.split(",").toList.map(_.trim.toInt))
    log.info("G1: " + g1.count())
    f1.foreach(println)

    val g = g0.cartesian(g1)
    log.info("g has %d flocks...".format(g.count()))

    //val f = f0.cartesian(f1)
    //println("f has %d flocks...".format(f.count()))
/*
    val MU = conf.mu()
    f.map(tuple => tuple._1.intersect(tuple._2).sorted)
      .filter(flock => flock.length >= MU)
      .distinct()
      .foreach(println)
*/
    // Saving results...
    PFlock.saveOutput()
    // Closing all...
    log.info(s"""{"content":"Closing app...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n""")
    simba.close()
  }

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val epsilon: ScallopOption[Double] = opt[Double](default = Some(100.0))
    val mu: ScallopOption[Int] = opt[Int](default = Some(3))
    val partitions: ScallopOption[Int] = opt[Int](default = Some(64))
    val cores: ScallopOption[Int] = opt[Int](default = Some(4))
    val master: ScallopOption[String] = opt[String](default = Some("local[*]"))
    val logs: ScallopOption[String] = opt[String](default = Some("INFO"))
    val path: ScallopOption[String] = opt[String](default = Some("Y3Q1/Datasets/"))
    val dataset: ScallopOption[String] = opt[String](default = Some("Berlin_N15K_A1K_T15"))
    val extension: ScallopOption[String] = opt[String](default = Some("csv"))
    verify()
  }
}
