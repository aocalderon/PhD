
object Cells {
  import org.apache.spark.sql.Row
  import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, LongType}
  import org.joda.time.{DateTime}
  import org.joda.time.format.DateTimeFormat

  /* ... new cell ... */

  val schema = StructType(
    Array(
      StructField("event", StringType, nullable=false),
      StructField("executor", IntegerType, nullable=false),
      StructField("task", IntegerType, nullable=false),
      StructField("start", LongType, nullable=false),
      StructField("end", LongType, nullable=false)
    )
  )
  val df = sparkSession.read.format("json").json("/opt/Logs/PFlock_4-28C/PFlock_4C")

  /* ... new cell ... */

  val tasks = df
    .filter(event => event.getString(5) == "SparkListenerTaskEnd")
    .select("event","Task Info.Executor ID","Task Info.Task ID","Task Info.Launch Time","Task Info.Finish Time")
    .map(event => (event.getString(0), event.getString(1), event.getLong(2)
                   , new DateTime(event.getLong(3)).toString("MM/dd/yyyy hh:mm:ss.SSS")
                   , new DateTime(event.getLong(4)).toString("MM/dd/yyyy hh:mm:ss.SSS")))
    .toDF("event","executor","task","start","end")

  /* ... new cell ... */

  TableChart(tasks.limit(100)) 

  /* ... new cell ... */
}
                  