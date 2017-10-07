import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.types.StructType
import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.slf4j.Logger
import org.slf4j.LoggerFactory

object FlockFinder {
  private val log: Logger = LoggerFactory.getLogger("myLogger")

  case class ST_Point(x: Double, y: Double, t: Int, id: Int)

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val epsilon: ScallopOption[Double] = opt[Double](default = Some(10.0))
    val mu: ScallopOption[Int] = opt[Int](default = Some(3))
    val partitions: ScallopOption[Int] = opt[Int](default = Some(64))
    val cores: ScallopOption[Int] = opt[Int](default = Some(4))
    val tstart: ScallopOption[Int] = opt[Int](default = Some(117))
    val tend: ScallopOption[Int] = opt[Int](default = Some(118))
    val cartesian_partitions: ScallopOption[Int] = opt[Int](default = Some(2))
    val master: ScallopOption[String] = opt[String](default = Some("local[*]"))
    val logs: ScallopOption[String] = opt[String](default = Some("INFO"))
    val path: ScallopOption[String] = opt[String](default = Some("Y3Q1/Datasets/"))
    val dataset: ScallopOption[String] = opt[String](default = Some("Berlin_N15K_A1K_T15"))
    val extension: ScallopOption[String] = opt[String](default = Some("csv"))
    verify()
  }
  
  def run(conf: Conf): Unit = {
    // Tuning master and number of cores...
    var MASTER = conf.master()
    if (conf.cores() == 1) {
      MASTER = "local"
    }
    // Setting parameters...
    val POINT_SCHEMA = ScalaReflection.schemaFor[ST_Point].
        dataType.
        asInstanceOf[StructType]
    // Starting a session...
    log.info("Setting paramaters...")
    val simba = SimbaSession.builder().
        master(MASTER).
        appName("FlockFinder").
        config("simba.index.partitions", s"${conf.partitions()}").
        config("spark.cores.max", s"${conf.cores()}").
        getOrCreate()
    simba.sparkContext.
        setLogLevel(conf.logs())
    // Calling implicits...
    import simba.implicits._
    import simba.simbaImplicits._
    val phd_home = scala.util.Properties.
        envOrElse("PHD_HOME", "/home/acald013/PhD/")
    val filename = s"$phd_home${conf.path()}${conf.dataset()}.${conf.extension()}"
    log.info("Reading %s ...".format(filename))
    val TSTART: Int = conf.tstart()
    val TEND: Int = conf.tend()
    val dataset = simba.
        read.option("header", "false").
        schema(POINT_SCHEMA).csv(filename).
        as[ST_Point].
        filter(datapoint => datapoint.t >= TSTART && datapoint.t <= TEND)
    dataset.cache()
    log.info("Number of points in dataset: %d".format(dataset.count()))
    var timestamps = dataset.
        map(datapoint => datapoint.t).
        distinct.
        sort("value").collect.toList
    // Setting MaximalFinder...
    MaximalFinder.EPSILON = conf.epsilon()
    MaximalFinder.MU = conf.mu()
    MaximalFinder.DATASET = conf.dataset()
    MaximalFinder.CORES = conf.cores()
    MaximalFinder.PARTITIONS = conf.partitions()
    // Running MaximalFinder...
    var timestamp = timestamps.head
    var currentPoints = dataset
        .filter(datapoint => datapoint.t == timestamp)
        .map(datapoint => 
            MaximalFinder.SP_Point(datapoint.id, datapoint.x, datapoint.y))
    log.info("%d points in time %d".format(currentPoints.count(), timestamp))
    val CARTESIAN_PARTITIONS: Int = conf.cartesian_partitions()
    var f0: RDD[List[Int]] = MaximalFinder.run(currentPoints, timestamp, simba).
        repartition(CARTESIAN_PARTITIONS)

    timestamps = timestamps.drop(1)
    for(timestamp <- timestamps){
        currentPoints = dataset
            .filter(datapoint => datapoint.t == timestamp)
            .map(datapoint => MaximalFinder.SP_Point(datapoint.id, datapoint.x, datapoint.y))
        log.info("%d points in time %d".format(currentPoints.count(), timestamp))
        var f1: RDD[List[Int]] = MaximalFinder.run(currentPoints, timestamp, simba).
            repartition(CARTESIAN_PARTITIONS)

        log.info("Running cartesian function between timestamps %d and %d...".format(timestamp, timestamp - 1))
        var combinations = f0.cartesian(f1)
        val ncombinations = combinations.count()
        log.info("Cartesian returns %d combinations...".format(ncombinations))

        val MU = conf.mu()
        val flocks = combinations.map{
                tuple => tuple._1.intersect(tuple._2).
                sorted
            }.
            filter(flock => flock.length >= MU).
            distinct()
        log.info("\n######\n#\n# Done!\n# %d flocks found in timestamp %d...\n#\n######".format(flocks.count(), timestamp))
    }
    // Saving results...
    MaximalFinder.saveOutput()
    // Closing all...
    log.info("Closing app...")
    simba.close()
  }

  def main(args: Array[String]): Unit = {
    // Setting a custom logger...
    log.info("Starting app...")
    // Reading arguments from command line...
    val conf = new Conf(args)
    FlockFinder.run(conf)
  }
}
