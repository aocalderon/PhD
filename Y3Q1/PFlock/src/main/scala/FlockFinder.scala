import scala.collection.mutable.ListBuffer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.simba.SimbaSession
import org.apache.spark.sql.types.StructType
import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.slf4j.Logger
import org.slf4j.LoggerFactory

object FlockFinder {
  private val log: Logger = LoggerFactory.getLogger("myLogger")
  private var phd_home: String = ""
  private var nPointset: Long = 0

  case class ST_Point(x: Double, y: Double, t: Int, id: Int)
  case class Flock(start: Int, end: Int, ids: List[Long], lon: Double = 0.0, lat: Double = 0.0)

  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val epsilon:    ScallopOption[Double] = opt[Double] (default = Some(10.0))
    val mu:         ScallopOption[Int]    = opt[Int]    (default = Some(5))
    val entries:    ScallopOption[Int]    = opt[Int]    (default = Some(25))
    val partitions: ScallopOption[Int]    = opt[Int]    (default = Some(1024))
    val candidates: ScallopOption[Int]    = opt[Int]    (default = Some(256))
    val cores:      ScallopOption[Int]    = opt[Int]    (default = Some(28))
    val master:     ScallopOption[String] = opt[String] (default = Some("spark://169.235.27.134:7077")) /* spark://169.235.27.134:7077 */
    val path:       ScallopOption[String] = opt[String] (default = Some("Y3Q1/Datasets/"))
    val valpath:    ScallopOption[String] = opt[String] (default = Some("Y3Q1/Validation/"))
    val dataset:    ScallopOption[String] = opt[String] (default = Some("Berlin_N15K_A1K_T15"))
    val extension:  ScallopOption[String] = opt[String] (default = Some("csv"))
    val method:     ScallopOption[String] = opt[String] (default = Some("fpmax"))
    // FlockFinder parameters
    val delta:	    ScallopOption[Int]    = opt[Int]    (default = Some(3))    
    val tstart:     ScallopOption[Int]    = opt[Int]    (default = Some(0))
    val tend:       ScallopOption[Int]    = opt[Int]    (default = Some(5))
    val cartesian:  ScallopOption[Int]    = opt[Int]    (default = Some(2))
    val logs:	    ScallopOption[String] = opt[String] (default = Some("INFO"))    
    verify()
  }
  
  def run(conf: Conf): Unit = {
    // Tuning master and number of cores...
    var MASTER = conf.master()
    if (conf.cores() == 1) {
      MASTER = "local"
    }
    // Setting parameters...
    val epsilon: Double = conf.epsilon()
    val mu: Int = conf.mu()
    val tstart: Int = conf.tstart()
    val tend: Int = conf.tend()
    val cartesian: Int = conf.cartesian()
    val partitions: Int = conf.partitions()
    val DELTA = conf.delta()
    val point_schema = ScalaReflection.schemaFor[ST_Point].
      dataType.
      asInstanceOf[StructType]
    // Starting a session...
    log.info("Setting paramaters...")
    val simba = SimbaSession.builder().
      master(MASTER).
      appName("FlockFinder").
      config("simba.index.partitions", s"${conf.partitions()}").
      getOrCreate()
    simba.sparkContext.setLogLevel(conf.logs())
    // Calling implicits...
    import simba.implicits._
    import simba.simbaImplicits._
    val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
    val filename = s"$phd_home${conf.path()}${conf.dataset()}.${conf.extension()}"
    log.info("Reading %s ...".format(filename))
    val pointset = simba.read.
      option("header", "false").
      schema(point_schema).
      csv(filename).
      as[ST_Point].
      filter(datapoint => datapoint.t >= tstart && datapoint.t <= tend).
      cache()
    nPointset = pointset.count()
    log.info("Number of points in dataset: %d".format(nPointset))
    var timestamps = pointset.
      map(datapoint => datapoint.t).
      distinct.
      sort("value").
      collect.toList
    var FLOCKS_OUT = List.empty[String]
    // Running experiment with different values of epsilon and mu...
    log.info("epsilon=%.1f,mu=%d".format(epsilon, mu))
    // Running MaximalFinder...
    var timestamp = timestamps.head
    var currentPoints = pointset
      .filter(datapoint => datapoint.t == timestamp)
      .map{ datapoint => 
        "%d,%.3f,%.3f".format(datapoint.id, datapoint.x, datapoint.y)
      }.
      rdd
    log.info("nPointset=%d,timestamp=%d".format(currentPoints.count(), timestamp))
    // Maximal disks for time 0...
    val maximals = MaximalFinderExpansion.
      run(currentPoints, simba, conf)
      
    ////////////////////////////////////////////////////////////////////
    saveStringArray(maximals.collect, "Flocks", conf)
    ////////////////////////////////////////////////////////////////////
      
    var F: RDD[Flock] = maximals.
      repartition(conf.cartesian()).
      map{ f => 
        Flock(timestamp, 
          timestamp, 
          f.split(";")(2).split(" ").toList.map(_.toLong)
        )
      }
    log.info("Flock,Start,End,Flock")
    // Reporting flocks for time 0...
    val FlockReport = F.map{ flock =>
      ("%d,%d,%s".
        format(flock.start, 
          flock.end, 
          flock.ids.mkString(" ")
        )
      )
    }.
    collect.
    mkString("\n")
    log.info(FlockReport)
    val n = F.filter(flock => flock.end - flock.start + 1 == DELTA).count()
    log.info("\n######\n#\n# Done!\n# %d flocks found in timestamp %d...\n#\n######".format(n, timestamp))
    
    // Maximal disks for time 1 and onwards
    for(timestamp <- timestamps.slice(1,timestamps.length)){
      // Reading points for current timestamp...
      currentPoints = pointset
        .filter(datapoint => datapoint.t == timestamp)
        .map{ datapoint => 
          "%d,%.3f,%.3f".format(datapoint.id, datapoint.x, datapoint.y)
        }.
        rdd
      log.info("nPointset=%d,timestamp=%d".format(currentPoints.count(), timestamp))
      
      // Finding maximal disks for current timestamp...
      val F_prime: RDD[Flock] = MaximalFinderExpansion.
        run(currentPoints, simba, conf).
        repartition(cartesian).
        map{ f => 
          Flock(timestamp, 
            timestamp, 
            f.split(";")(2).split(" ").toList.map(_.toLong)
          )
        }
      // Joining previous flocks and current ones...
      log.info("Running cartesian function for timestamps %d...".format(timestamp))
      var combinations = F.cartesian(F_prime)

      //////////////////////////////////////////////////////////////////
      log.info("\nPrinting F... %s".format(printFlocks(F)))
      log.info("\nPrinting F_prime... %s".format(printFlocks(F_prime)))
      //////////////////////////////////////////////////////////////////

      val ncombinations = combinations.count()
      log.info("Cartesian returns %d combinations...".format(ncombinations))
      // Checking if ids intersect...
      var F_temp = combinations.
        map{ tuple => 
          val s = tuple._1.start
          val e = tuple._2.end
          val ids1 = tuple._1.ids
          val ids2 = tuple._2.ids
          val ids_in_common = ids1.intersect(ids2).sorted
          Flock(s, e, ids_in_common)
        }.
        // Checking if they are greater than mu...
        filter(flock => flock.ids.length >= mu).
        // Removing duplicates...
        distinct

      //////////////////////////////////////////////////////////////////
      F_temp = F_temp.mapPartitions{ records =>
        var flocks = new ListBuffer[(Flock, Boolean)]()
        for(record <- records){
          flocks += Tuple2(record, true)
        }
        for(i <- 0 until flocks.length){ 
          for(j <- 0 until flocks.length){ 
            if(i != j & flocks(i)._2){
              val ids1 = flocks(i)._1.ids
              val ids2 = flocks(j)._1.ids
              println("%d -> (%s:%s) <=> %d -> (%s:%s)".format(
                i, ids1.mkString(" "), flocks(i)._2.toString, 
                j, ids2.mkString(" "), flocks(j)._2.toString)
              ) 
              if(flocks(j)._2 & ids1.forall(ids2.contains)){
                flocks(i) = Tuple2(flocks(i)._1, false)
              }
              if(flocks(i)._2 & ids2.forall(ids1.contains)){
                flocks(j) = Tuple2(flocks(j)._1, false)
              }
            }
          }
        }
        flocks.filter(_._2).map(_._1).toIterator
      }
      
      F_temp = F_temp.repartition(1).
        mapPartitions{ records =>
          var flocks = new ListBuffer[(Flock, Boolean)]()
          for(record <- records){
            flocks += Tuple2(record, true)
          }
          for(i <- 0 until flocks.length){ 
            for(j <- 0 until flocks.length){ 
              if(i != j & flocks(i)._2){
                val ids1 = flocks(i)._1.ids
                val ids2 = flocks(j)._1.ids
                if(flocks(j)._2 & ids1.forall(ids2.contains)){
                  flocks(i) = Tuple2(flocks(i)._1, false)
                }
                if(flocks(i)._2 & ids2.forall(ids1.contains)){
                  flocks(j) = Tuple2(flocks(j)._1, false)
                }
              }
            }
          }
          flocks.filter(_._2).map(_._1).toIterator
      }.
      repartition(partitions)
      //////////////////////////////////////////////////////////////////
        
      //////////////////////////////////////////////////////////////////
      log.info("\nPrinting F_temp... %s".format(printFlocks(F_temp)))
      //////////////////////////////////////////////////////////////////

      
      // Reporting flocks with delta duration...
      /*
      val FlockReports = F.map{ flock =>
        ("%d,%d,%s".format(flock.start, flock.end, flock.ids.mkString(" ")))
      }.collect.mkString("\n")
      log.info(FlockReports)
      */
      // Reporting the number of flocks...
      val n = F_temp.filter(flock => flock.end - flock.start + 1 == DELTA).count()
      log.info("\n######\n#\n# Done!\n# %d flocks found in timestamp %d...\n#\n######".format(n, timestamp))
      // Appending new potential flocks from current timestamp...
      F = F_temp
    }
    // Closing all...
    log.info("Closing app...")
    simba.close()
  }
  
  def saveStringArray(array: Array[String], tag: String, conf: Conf): Unit = {
    val path = s"$phd_home${conf.valpath()}"
    val filename = s"${conf.dataset()}_E${conf.epsilon()}_M${conf.mu()}_C${conf.cores()}"
    new java.io.PrintWriter("%s%s_%s_%d.txt".format(path, filename, tag, System.currentTimeMillis)) {
      write(array.mkString("\n"))
      close()
    }
  }
  
  def printFlocks(flocks: RDD[Flock]): String ={
    val n = flocks.count()
    val info = flocks.map{ f => 
      "\n%d,%d,%s".format(f.start, f.end, f.ids.mkString(" "))
    }.collect.mkString("")
    
    "# of flocks: %d\n%s".format(n,info)
  }

  def main(args: Array[String]): Unit = {
    phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/acald013/PhD/")
    log.info("Starting app...")
    // Reading arguments from command line...
    val conf = new Conf(args)
    FlockFinder.run(conf)
  }
}
