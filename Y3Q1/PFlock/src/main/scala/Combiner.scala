import org.apache.spark.sql.simba.SimbaSession

object Combiner {
  val MU: Int = 3

  def main(args: Array[String]): Unit = {
    val simba = SimbaSession
      .builder()
      .master("local[*]")
      .appName("Combiner")
      .config("simba.index.partitions", "64")
      .config("spark.cores.max", "4")
      .getOrCreate()
    simba.sparkContext.setLogLevel("ERROR")
    val f0 = simba.sparkContext
      .textFile("file:///home/and/Documents/PhD/Code/Y3Q1/Datasets/s1.txt")
      .map(_.split(",").toList.map(_.trim.toInt))
    println("F0: " + f0.count())
    f0.foreach(println)
    val f1 = simba.sparkContext
      .textFile("file:///home/and/Documents/PhD/Code/Y3Q1/Datasets/s2.txt")
      .map(_.split(",").toList.map(_.trim.toInt))
    println("F1: " + f1.count())
    f1.foreach(println)

    val f = f0.cartesian(f1)
    println(f.count())

    f.map(tuple => tuple._1.intersect(tuple._2).sorted)
      .filter(flock => flock.length >= MU)
      .distinct()
      .foreach(println)


    simba.close()
  }
}
