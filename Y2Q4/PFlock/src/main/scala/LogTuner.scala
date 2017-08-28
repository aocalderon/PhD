import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}

import org.apache.spark.sql.SparkSession
import org.joda.time.DateTime
import org.rogach.scallop.{ScallopConf, ScallopOption}

object LogTuner {
  def main(args: Array[String]): Unit = {
    val conf = new Conf(args)
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("LogTuner")
      .config("spark.sql.warehouse.dir", "file:/tmp/spark-warehouse")
      .config("spark.eventLog.enabled","false")
      .getOrCreate()

    import spark.implicits._
    spark.sparkContext.setLogLevel(conf.logs())

    val df = spark.read.format("json")
      .json(s"${conf.dirlogs()}/${conf.filename()}")
      .filter("id is not null")
      .select("id","content","start","end","group","type")
    val temp = df.filter("content = 'Reading data...'").select("start").collect
    val start = new DateTime(temp(0).getString(0).replace(" ", "T")).getMillis
    println(start)
    df.map{row =>
      val id = row.getLong(0)
      val content = row.getString(1)
      val oldstart = new DateTime(row.getString(2).replace(" ", "T")).getMillis
      val newstart = new DateTime(oldstart - start).toString("yyyy-MM-dd hh:mm:ss.SSS")
      val newend = ""
      if(row.getString(3) != null) {
        val oldend = new DateTime(row.getString(3).replace(" ", "T")).getMillis
        val newend = new DateTime(oldend - start).toString("yyyy-MM-dd hh:mm:ss.SSS")
      }
      val group = row.getLong(4)
      val ttype = row.getString(5)
      (id, content, newstart, newend, group, ttype)
    }.show

    /*
    val jsonname = s"${conf.dirlogs()}/tuned-${conf.filename()}.json"
    val json = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(jsonname)))
    json.write("[\n")
    json.write(df.toJSON.collect.mkString(",\n"))
    json.write("\n]")
    json.close()
    */
  }

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val filename: ScallopOption[String] = opt[String](default = Some("data9.json"))
    val dirlogs: ScallopOption[String] = opt[String](default = Some("/opt/Spark/Logs/logs"))
    val logs: ScallopOption[String] = opt[String](default = Some("ERROR"))
    verify()
  }

}
