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
      .textFile("file:///home/and/Documents/PhD/Code/Y3Q1/Datasets/f0.txt")
      .map(_.split(",").toList.map(_.toInt))
    println(f0.count())
    val f1 = simba.sparkContext
      .textFile("file:///home/and/Documents/PhD/Code/Y3Q1/Datasets/f1.txt")
      .map(_.split(",").toList.map(_.toInt))
    println(f1.count())
    val f = f0.cartesian(f1)
    println(f.count())

    f.map(tuple => tuple._1.intersect(tuple._2).sorted)
      .filter(flock => flock.length >= MU)
      .distinct()
      .foreach(println)

    simba.close()
  }
}
