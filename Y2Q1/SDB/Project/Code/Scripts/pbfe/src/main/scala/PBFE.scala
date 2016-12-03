import org.apache.spark.sql.{Point, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

object PBFE {

  case class PointItem(id: Int, x: Double, y: Double)

  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("PBFE")
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    sc.setLogLevel("ERROR")
    sqlContext.setConf("spark.sql.shuffle.partitions", 4.toString)
    sqlContext.setConf("spark.sql.sampleRate", 1.toString)
    sqlContext.setConf("spark.sql.partitioner.strTransferThreshold", 1000000.toString)

    import sqlContext.implicits._

    val filename = args(0)
    val epsilon = args(1).toInt
    val tag = filename.substring(filename.lastIndexOf("/") + 1).split("\\.")(0).substring(1)

	val p1 = sc.textFile(filename).map(_.split(",")).map(p => PointItem(p(0).trim.toInt, p(1).trim.toDouble, p(2).trim.toDouble)).toDF()
    val p2 = p1.toDF("id2", "x2", "y2")


    val time1 = System.currentTimeMillis()
    val pairs = p1.distanceJoin(p2, Point(p1("x"), p1("y")), Point(p2("x2"), p2("y2")), epsilon)
    val disks = pairs.rdd.filter(pair => (pair(0)).toString.toInt > (pair(3)).toString.toInt )
    val time2 = System.currentTimeMillis()
    println("PBFE," + epsilon + "," + tag  + "," + disks.count()*2 + "," + (time2 - time1) / 1000.0)

    sc.stop()
  }
}

