import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}

import SPMF.AlgoFPMax
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.simba.index.RTreeType
import org.apache.spark.sql.simba.{Dataset, SimbaSession}

import scala.collection.JavaConverters._

/**
  * Created by and on 5/4/17.
  */
object PFlock {
  // Setting variables...
  var EPSILON: Double = 10.0
  var MU: Int = 3
  var DATASET: String = "Berlin"
  var CORES: Int = 0
  var PARTITIONS: Int = 0
  var LOG = List.empty[String]
  var OUTPUT = List.empty[String]
  private val DELTA: Double = 0.01

  case class SP_Point(id: Int, x: Double, y: Double)
  case class ACenter(id: Long, x: Double, y: Double)
  case class BBox(minx: Double, miny: Double, maxx: Double, maxy: Double)

  def run(points: Dataset[SP_Point]
          , timestamp: Int
          , simba: SimbaSession
          , epsilon: Double = PFlock.EPSILON
          , mu: Int = PFlock.MU): RDD[List[Int]] ={
    // Calling implicits...
    import simba.implicits._
    import simba.simbaImplicits._
    // Starting timer...
    LOG = LOG :+ s"""{"content":"Indexing points...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
    var time1: Long = System.currentTimeMillis()
    // Indexing points...
    val p1 = points.toDF("id1", "x1", "y1")
    p1.index(RTreeType, "p1RT", Array("x1", "y1"))
    val p2 = points.toDF("id2", "x2", "y2")
    p2.index(RTreeType, "p2RT", Array("x2", "y2"))
    // Self-joining to find pairs of points close enough (< epsilon)...
    LOG = LOG :+ s"""{"content":"Finding pairs (Self-join)...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
    val pairsRDD = p1.distanceJoin(p2, Array("x1", "y1"), Array("x2", "y2"), epsilon).rdd
    // Calculating disk's centers coordinates...
    LOG = LOG :+ s"""{"content":"Computing disks...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
    val centers = findDisks(pairsRDD, epsilon)
      .distinct()
      .toDS()
      .index(RTreeType, "centersRT", Array("x", "y"))
      .withColumn("id", monotonically_increasing_id())
      .as[ACenter]
    // Grouping objects enclosed by candidates disks...
    LOG = LOG :+ s"""{"content":"Mapping disks and points...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
    val candidates = centers
      .distanceJoin(p1, Array("x", "y"), Array("x1", "y1"), (epsilon / 2) + DELTA)
      .groupBy("id", "x", "y")
      .agg(collect_list("id1").alias("IDs"))
    val ncandidates = candidates.count()
    var time2: Long = System.currentTimeMillis()
    val timeD: Double = (time2 - time1) / 1000.0
    // Filtering candidates less than mu...
    time1 = System.currentTimeMillis()
    LOG = LOG :+ s"""{"content":"Filtering less-than-mu disks...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
    val filteredCandidates =  candidates.filter(row => row.getList(3).size() >= mu)
      .map(d => (d.getLong(0), d.getDouble(1), d.getDouble(2), d.getList[Integer](3).toString))
    var nmaximal: Long = 0
    var maximal: RDD[List[Int]] = simba.sparkContext.emptyRDD
    // Prevent indexing of empty collections...
    if(filteredCandidates.count() != 0){
      // Indexing remaining candidates disks...
      filteredCandidates.index(RTreeType, "candidatesRT", Array("_2", "_3"))
      // Filtering redundant candidates
      LOG = LOG :+ s"""{"content":"Getting maximals inside...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
      val maximalInside = filteredCandidates
        .rdd
        .mapPartitions { partition =>
          val transactions = partition
            .map { disk =>
              disk._4
                .replace("[","")
                .replace("]","")
                .split(",")
                .map{ id =>
                  new Integer(id.trim)
                }
                .sorted
                .toList
                .asJava
            }.toList.asJava
          val fpMax = new AlgoFPMax
          val itemsets = fpMax.runAlgorithm(transactions, 1)
          itemsets.getItemsets(mu).asScala.toIterator
        }
      maximalInside.count()
      LOG = LOG :+ s"""{"content":"Getting maximals in frame...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
      val maximalFrame = filteredCandidates
        .rdd
        .mapPartitions { partition =>
          val pList = partition.toList
          val bbox = getBoundingBox(pList)
          val transactions = pList
            .map(disk => (disk._1, disk._2, disk._3, disk._4, !isInside(disk._2, disk._3, bbox, epsilon)))
            .filter(_._5)
            .map { disk =>
              disk._4
                .replace("[","")
                .replace("]","")
                .split(",")
                .map{ id =>
                  new Integer(id.trim)
                }
                .sorted
                .toList
                .asJava
            }.asJava
          val fpMax = new AlgoFPMax
          val itemsets = fpMax.runAlgorithm(transactions, 1)
          itemsets.getItemsets(mu).asScala.toIterator
        }
      maximalFrame.count()
      LOG = LOG :+ s"""{"content":"Prunning duplicates...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
      maximal = maximalInside.union(maximalFrame).distinct().map(_.asScala.toList.map(_.intValue()))
      nmaximal = maximal.count()
    }
    // Stopping timer...
    time2 = System.currentTimeMillis()
    val timeM: Double = (time2 - time1) / 1000.0
    val time: Double = BigDecimal(timeD + timeM).setScale(3, BigDecimal.RoundingMode.HALF_DOWN).toDouble
    // Print summary...
    val record = s"PFlock,$epsilon,$timestamp,$timeD,$timeM,$time,$ncandidates,$nmaximal,$CORES,$PARTITIONS,${org.joda.time.DateTime.now.toLocalTime}\n"
    OUTPUT = OUTPUT :+ record
    print("%10.10s %10.1f %10.10s %10.3f %10.3f %10.3f %10d %10d %10d %10d %15.15s\n"
      .format("PFlock",epsilon,timestamp,timeD,timeM,time,ncandidates,nmaximal,CORES,PARTITIONS,org.joda.time.DateTime.now.toLocalTime))
    // Dropping indices
    LOG = LOG :+ s"""{"content":"Dropping indices...","start":"${org.joda.time.DateTime.now.toLocalDateTime}"},\n"""
    p1.dropIndexByName("p1RT")
    p2.dropIndexByName("p2RT")
    centers.dropIndexByName("centersRT")
    filteredCandidates.dropIndexByName("candidatesRT")

    maximal
  }
  
  def findDisks(pairsRDD: RDD[Row], epsilon: Double): RDD[ACenter] = {
    val r2: Double = math.pow(epsilon / 2, 2)
    val centers = pairsRDD
      .filter((row: Row) => row.getInt(0) != row.getInt(3))
      .map { (row: Row) =>
        val p1 = SP_Point(row.getInt(0), row.getDouble(1), row.getDouble(2))
        val p2 = SP_Point(row.getInt(3), row.getDouble(4), row.getDouble(5))
        calculateDiskCenterCoordinates(p1, p2, r2)
      }
    centers
  }

  def calculateDiskCenterCoordinates(p1: SP_Point, p2: SP_Point, r2: Double): ACenter = {
    val X: Double = p1.x - p2.x
    val Y: Double = p1.y - p2.y
    var D2: Double = math.pow(X, 2) + math.pow(Y, 2)
    if (D2 == 0)
      D2 = 0.01
    val root: Double = math.sqrt(math.abs(4.0 * (r2 / D2) - 1.0))
    val h1: Double = ((X + Y * root) / 2) + p2.x
    val k1: Double = ((Y - X * root) / 2) + p2.y

    ACenter(0, h1, k1)
  }

  def isInside(x: Double, y: Double, bbox: BBox, epsilon: Double): Boolean ={
    x < (bbox.maxx - epsilon) &&
      x > (bbox.minx + epsilon) &&
      y < (bbox.maxy - epsilon) &&
      y > (bbox.miny + epsilon)
  }

  def getBoundingBox(p: List[(Long, Double, Double, Any)]): BBox = {
    var minx: Double = Double.MaxValue
    var miny: Double = Double.MaxValue
    var maxx: Double = Double.MinValue
    var maxy: Double = Double.MinValue
    p.foreach{r =>
      if(r._2 < minx){
        minx = r._2
      }
      if (r._2 > maxx){
        maxx = r._2
      }
      if(r._3 < miny){
        miny = r._3
      }
      if(r._3 > maxy){
        maxy = r._3
      }
    }
    BBox(minx, miny, maxx, maxy)
  }

  def toWKT(bbox: BBox): String = "POLYGON (( %f %f, %f %f, %f %f, %f %f, %f %f ))"
    .format(
      bbox.minx, bbox.maxy,
      bbox.maxx, bbox.maxy,
      bbox.maxx, bbox.miny,
      bbox.minx, bbox.miny,
      bbox.minx, bbox.maxy
    )

  def toWKT(x: Double, y: Double): String = "POINT (%f %f)".format(x, y)

  def saveOutput(): Unit ={
    val output = s"${DATASET}_E${EPSILON}_M${MU}_C${CORES}_P${PARTITIONS}_${System.currentTimeMillis()}.csv"
    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output)))
    OUTPUT.foreach(writer.write)
    writer.close()
  }
}
