/*
 * Copyright 2016 by Simba Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import java.net.URL

import edu.utah.cs.simba.SimbaContext
import edu.utah.cs.simba.index.RTreeType
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer
import ca.pfv.spmf.algorithms.frequentpatterns.lcm.{Dataset, AlgoLCM}
import ca.pfv.spmf.patterns.itemset_array_integers_with_count.Itemsets

/**
  * Created by dongx on 11/14/2016.
  */
object TestMain {
  case class PointData(x: Double, y: Double, z: Double, other: String)

  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("SpatialOperationExample").setMaster("local[4]")
    val sc = new SparkContext(sparkConf)
    val simbaContext = new SimbaContext(sc)

    var leftData = ListBuffer[PointData]()
    var rightData = ListBuffer[PointData]()

    import simbaContext.implicits._
    import simbaContext.SimbaImplicits._

    for (i <- 1 to 1000){
      leftData += PointData( i + 0.0, i + 0.0, i + 0.0, "a = " + i)
      rightData += PointData(i + 0.0, i + 0.0, i + 0.0, "a = " + (i + 1))
    }

    val leftDF = sc.parallelize(leftData).toDF
    val rightDF = sc.parallelize(rightData).toDF

    //leftDF.registerTempTable("point1")

    //simbaContext.sql("SELECT * FROM point1 WHERE x < 10").collect().foreach(println)

    //    simbaContext.indexTable("point1", RTreeType, "rt", List("x", "y"))
    leftDF.index(RTreeType, "rt", Array("x", "y"))

    val df = leftDF.distanceJoin(rightDF, Array("x", "y"), Array("x", "y"), 10.0)
    println(df.queryExecution)
    df.show()

    val filename = "/opt/Java/IdeaProjects/SPMF/src/ca/pfv/spmf/algorithms/frequentpatterns/lcm/contextPasquier99.txt"
    val minsup = 0.4
    val input = java.net.URLDecoder.decode(filename,"UTF-8")
    val dataset = new Dataset(input)
    val algo = new AlgoLCM()
    val itemsets = algo.runAlgorithm(minsup, dataset, null)

    algo.printStats();
    itemsets.printItemsets(dataset.getTransactions().size());

    //    leftDF.range(Array("x", "y"), Array(4.0, 5.0), Array(111.0, 222.0)).show(100)
    //    leftDF.knnJoin(rightDF, Array("x", "y"), Array("x", "y"), 3).show(100)

    sc.stop()
  }
}