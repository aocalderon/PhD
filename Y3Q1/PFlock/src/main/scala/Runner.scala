import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}

import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.types.StructType
import org.rogach.scallop.{ScallopConf, ScallopOption}

object Runner {
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
    val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/and/Documents/PhD/Code/")
    val filename = s"$phd_home${conf.path()}${conf.dataset()}.${conf.extension()}"
    println(filename)
    val dataset = simba.read
      .option("header", "false")
      .schema(POINT_SCHEMA)
      .csv(filename)
      .as[APoint]
      .filter(datapoint => datapoint.t < 119)
    dataset.show()
    println(dataset.count())
    val timestamps = dataset.map(datapoint => datapoint.t).distinct.sort("value").collect
    timestamps.foreach(println)
    // Setting PFlock...
    PFlock.MASTER = conf.master()
    PFlock.CORES = conf.cores()
    PFlock.EPSILON = conf.epsilon()
    PFlock.MU = conf.mu()
    PFlock.DATASET = conf.dataset()
    PFlock.PARTITIONS = conf.partitions()
    // Running PFlock...
    timestamps.foreach{ timestamp =>
      val points = dataset
    		.filter(datapoint => datapoint.t == timestamp)
    		.map(datapoint => PFlock.SP_Point(datapoint.id, datapoint.x, datapoint.y))
      println("%d points in time %d".format(points.count(), timestamp))
      PFlock.run(points, timestamp, 10.0, 3, simba)
    }
    // Saving results...
    val output = s"${PFlock.DATASET}_E${PFlock.EPSILON}_M${PFlock.MU}_C${PFlock.CORES}_P${PFlock.PARTITIONS}_${System.currentTimeMillis()}.csv"
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output)))
    PFlock.OUTPUT.foreach(writer.write)
    writer.close()
    
    simba.close()
  }

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val epsilon: ScallopOption[Double] = opt[Double](default = Some(10.0))
    val mu: ScallopOption[Int] = opt[Int](default = Some(3))
    val partitions: ScallopOption[Int] = opt[Int](default = Some(64))
    val cores: ScallopOption[Int] = opt[Int](default = Some(4))
    val master: ScallopOption[String] = opt[String](default = Some("local[*]"))
    val logs: ScallopOption[String] = opt[String](default = Some("ERROR"))
    val phd_home: ScallopOption[String] = opt[String](default = sys.env.get("PHD_HOME"))
    val path: ScallopOption[String] = opt[String](default = Some("Y3Q1/Datasets/"))
    val dataset: ScallopOption[String] = opt[String](default = Some("Berlin_N277K_A18K_T15"))
    val extension: ScallopOption[String] = opt[String](default = Some("csv"))

    verify()
  }

}
