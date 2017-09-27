import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}

import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.joda.time.DateTime
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

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
      .map(event => (s"Task ${event.getLong(2)}"
        , event.getString(1).toInt + 1
        //, event.getLong(2)
        , new DateTime(event.getLong(3)).toString("yyyy-MM-dd hh:mm:ss.SSS")
        , new DateTime(event.getLong(4)).toString("yyyy-MM-dd hh:mm:ss.SSS")
        , "point"))
      .toDF("content","group","start","end","type")
    val nstages = df.select("Executor ID").distinct().count
    val stages = df
      .filter(event => event.getString(5) == "SparkListenerStageCompleted")
      .select("event","Stage Info.Stage Name","Stage Info.Details","Stage Info.Stage ID","Stage Info.Submission Time","Stage Info.Completion Time")
      .map(event => (s"${event.getString(1)}}"
        , nstages
        //, event.getLong(3)
        , new DateTime(event.getLong(4)).toString("yyyy-MM-dd hh:mm:ss.SSS")
        , new DateTime(event.getLong(5)).toString("yyyy-MM-dd hh:mm:ss.SSS")
        , "range"))
      .toDF("content","group","start","end","type")
    //stages.show()
    val njobs= nstages + 1
    val jobs = df
      .filter(event => event.getString(5) == "SparkListenerJobEnd")
      .select("Job ID","Completion Time")
      .join(df
        .filter(event => event.getString(5) == "SparkListenerJobStart")
        .select("Job ID","Submission Time"), "Job ID")
      .map(event => (s" Job ${event.getLong(0)}"
        , njobs
        //, event.getLong(0)
        , new DateTime(event.getLong(2)).toString("yyyy-MM-dd hh:mm:ss.SSS")
        , new DateTime(event.getLong(1)).toString("yyyy-MM-dd hh:mm:ss.SSS")
        , "range"))
      .toDF("content","group","start","end","type")
    //jobs.show()

    val parts = spark.read.format("json").json(s"${conf.dirlogs()}/${conf.filename()}.json")
      .map(event => (event.getString(0)
        , 0
        , new DateTime(event.getString(1)).toString("yyyy-MM-dd hh:mm:ss.SSS")))
      .toDF("content","group","start")
      .withColumn("end", lit(null: String))
      .withColumn("type", lit(null: String))
    val data = parts.union(tasks.union(stages.union(jobs)))
      .withColumn("id", monotonically_increasing_id)
    //data.repartition(1)
      //.write
      //.json(s"${conf.dirlogs()}/timeline-${conf.filename()}")

    val jsonname = s"${conf.dirlogs()}/timeline-${conf.filename()}.json"
    val json = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(jsonname)))
    json.write("[\n")
    json.write(data.toJSON.collect.mkString(",\n"))
    json.write("\n]")
    json.close()

  }

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val filename: ScallopOption[String] = opt[String](default = Some("app-20170806215838-0006"))
    val dirlogs: ScallopOption[String] = opt[String](default = Some("/opt/Spark/Logs"))
    val logs: ScallopOption[String] = opt[String](default = Some("ERROR"))
    verify()
  }
}
