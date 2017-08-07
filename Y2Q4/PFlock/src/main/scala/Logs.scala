import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, LongType}
import org.joda.time.{DateTime}
import org.joda.time.format.DateTimeFormat
import org.apache.spark.sql.SparkSession

object Logs {
  def main(args: Array[String]): Unit = {
    val conf = new Conf(args)
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("Logs")
      .config("spark.sql.warehouse.dir", "file:/tmp/spark-warehouse")
      .config("spark.eventLog.enabled","false")
      .getOrCreate()

    import spark.implicits._
    spark.sparkContext.setLogLevel(conf.logs())
    val df = spark.read.format("json").json(s"${conf.dirlogs()}/${conf.filename()}")
    val tasks = df
      .filter(event => event.getString(5) == "SparkListenerTaskEnd")
      .select("event","Task Info.Executor ID","Task Info.Task ID","Task Info.Launch Time","Task Info.Finish Time")
      .map(event => (s"Task ${event.getLong(2)}", event.getString(1), event.getLong(2)
        , new DateTime(event.getLong(3)).toString("MM/dd/yyyy hh:mm:ss.SSS")
        , new DateTime(event.getLong(4)).toString("MM/dd/yyyy hh:mm:ss.SSS")
        , "point"))
      .toDF("content","group","id","start","end","type")
    tasks.show()
    val stages = df
      .filter(event => event.getString(5) == "SparkListenerStageCompleted")
      .select("event","Stage Info.Stage Name","Stage Info.Details","Stage Info.Stage ID","Stage Info.Submission Time","Stage Info.Completion Time")
      .map(event => (s"${event.getString(1)} ${event.getString(2)}", event.getLong(3)
        , new DateTime(event.getLong(4)).toString("MM/dd/yyyy hh:mm:ss.SSS")
        , new DateTime(event.getLong(5)).toString("MM/dd/yyyy hh:mm:ss.SSS")
        , "range"))
      .toDF("content","id","start","end","type")
    stages.show()
  }

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val filename: ScallopOption[String] = opt[String](default = Some("local-1502073502069"))
    val dirlogs: ScallopOption[String] = opt[String](default = Some("/opt/Spark/Logs"))
    val logs: ScallopOption[String] = opt[String](default = Some("ERROR"))
    verify()
  }
}
