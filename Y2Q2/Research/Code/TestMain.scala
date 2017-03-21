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
import java.util

import edu.utah.cs.simba.SimbaContext
import edu.utah.cs.simba.index.RTreeType
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer
import ca.pfv.spmf.algorithms.frequentpatterns.lcm.{AlgoLCM, Dataset}
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

    val minsup = 0.4
    val t1 = new util.ArrayList[Integer]()
    t1.add(1)
    t1.add(Integer.valueOf(2))
    t1.add(Integer.valueOf(5))
    t1.add(Integer.valueOf(7))
    t1.add(Integer.valueOf(9))
    val t2 = new util.ArrayList[Integer]()
    t2.add(Integer.valueOf(1))
    t2.add(Integer.valueOf(3))
    t2.add(Integer.valueOf(5))
    t2.add(Integer.valueOf(7))
    t2.add(Integer.valueOf(9))
    val t3 = new util.ArrayList[Integer]()
    t3.add(Integer.valueOf(1))
    t3.add(Integer.valueOf(4))
    t3.add(Integer.valueOf(1))
    t3.add(Integer.valueOf(3))
    val t4 = new util.ArrayList[Integer]()
    t4.add(Integer.valueOf(1))
    t4.add(Integer.valueOf(3))
    t4.add(Integer.valueOf(4))
    t4.add(Integer.valueOf(5))
    t4.add(Integer.valueOf(6))
    val t5 = new util.ArrayList[Integer]()
    t5.add(Integer.valueOf(1))
    t5.add(Integer.valueOf(2))
    val t6 = new util.ArrayList[Integer]()
    t6.add(Integer.valueOf(2))
    t6.add(Integer.valueOf(1))
    val t7 = new util.ArrayList[Integer]()
    t7.add(Integer.valueOf(1))
    t7.add(Integer.valueOf(7))
    t7.add(Integer.valueOf(2))
    t7.add(Integer.valueOf(3))
    t7.add(Integer.valueOf(4))
    t7.add(Integer.valueOf(5))
    t7.add(Integer.valueOf(8))
    t7.add(Integer.valueOf(9))
    val t8 = new util.ArrayList[Integer]()
    t8.add(Integer.valueOf(6))
    t8.add(Integer.valueOf(1))
    t8.add(Integer.valueOf(2))
    val t9 = new util.ArrayList[Integer]()
    t9.add(Integer.valueOf(4))
    t9.add(Integer.valueOf(5))
    t9.add(Integer.valueOf(6))
    val t10 = new util.ArrayList[Integer]()
    t10.add(Integer.valueOf(8))
    t10.add(Integer.valueOf(2))
    t10.add(Integer.valueOf(5))
    val t11 = new util.ArrayList[Integer]()
    t11.add(Integer.valueOf(9))
    t11.add(Integer.valueOf(2))
    t11.add(Integer.valueOf(1))
    val t12 = new util.ArrayList[Integer]()
    t12.add(Integer.valueOf(1))
    t12.add(Integer.valueOf(2))
    t12.add(Integer.valueOf(4))
    t12.add(Integer.valueOf(8))
    t12.add(Integer.valueOf(9))
    val ts = new util.ArrayList[util.ArrayList[Integer]]
    ts.add(t1)
    ts.add(t2)
    ts.add(t3)
    ts.add(t4)
    ts.add(t5)
    ts.add(t6)
    ts.add(t7)
    ts.add(t8)
    ts.add(t9)
    ts.add(t10)
    ts.add(t11)
    ts.add(t12)
    val dataset = new Dataset(ts)
    val algo = new AlgoLCM
    val itemsets = algo.runAlgorithm(minsup, dataset, null.asInstanceOf[String])
    algo.printStats()
    itemsets.printItemsets(dataset.getTransactions.size)

    //    leftDF.range(Array("x", "y"), Array(4.0, 5.0), Array(111.0, 222.0)).show(100)
    //    leftDF.knnJoin(rightDF, Array("x", "y"), Array("x", "y"), 3).show(100)

    sc.stop()
  }
}