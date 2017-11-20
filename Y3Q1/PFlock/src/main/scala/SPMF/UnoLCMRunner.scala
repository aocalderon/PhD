package SPMF

import java.io.PrintWriter
import scala.collection.mutable
import scala.io.Source
import scala.language.postfixOps
import sys.process._

object UnoLCMRunner {
    def main(args: Array[String]): Unit = {
        var data = new mutable.HashSet[(Int, String)]
        val lines = Source.fromFile("/home/acald013/tmp/DB80K_E80.0_M25_Maximals.txt")
        println("Reading file...")
        for(line <- lines.getLines()){
            val IdAndTransaction = line.split(";")
            val tid = IdAndTransaction(0).toInt
            val transaction = IdAndTransaction(1).split(" ").map(_.toInt).sorted.toList.mkString(" ")
            data += (tid -> transaction)
        }
        lines.close()
        println("Running LCM...")
        val transactionsByTID = data.groupBy(x => x._1).mapValues(v => v.map(_._2).mkString("\n"))
        for(tid <- transactionsByTID.keys){
            val output = transactionsByTID.get(tid).toString
            new PrintWriter("/tmp/input%d.txt".format(tid)){
                write(output)
                close()
            }
            println("lcm M /tmp/input%d.txt 1 /tmp/output%d.txt".format(tid, tid))
            "time lcm M /tmp/input%d.txt 1 /tmp/output%d.txt".format(tid,tid) !
        }
    }
}

