package SPMF

import java.io.PrintWriter
import scala.collection.mutable
import scala.io.Source
import scala.language.postfixOps
import sys.process._

object UnoLCMRunner {
    def main(args: Array[String]): Unit = {
        var data = new mutable.HashSet[(Int, Set[Int])]
        val lines = Source.fromFile("/home/acald013/tmp/DB80K_E100.0_M50_Maximals.txt")
        println("Reading file...")
	for(line <- lines.getLines()){
            val IdAndTransaction = line.split(";")
            val tid = IdAndTransaction(0).toInt
            val transaction = IdAndTransaction(1).split(" ").map(_.toInt).toList.sorted.toSet
            data += (tid -> transaction)
        }
        lines.close()
	println("Running LCM...")
        val transactionsByTID = data.groupBy(x => x._1).mapValues(v => v.map(m => m._2))
        for(tid <- transactionsByTID.keys){
            val output = transactionsByTID.get(tid).map(t => t.mkString(" ")).toList.mkString("\n")
            new PrintWriter("/tmp/input%d.txt".format(tid)){
                write(output)
                close()
            }
            println("lcm M /tmp/input%d.txt 1 -".format(tid))
            "lcm M /tmp/input%d.txt 1 -".format(tid) !
        }
    }
}

