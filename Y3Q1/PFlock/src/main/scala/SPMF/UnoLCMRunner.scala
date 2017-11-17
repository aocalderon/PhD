package SPMF

import java.io.PrintWriter
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable
import scala.io.Source
import scala.language.postfixOps
import sys.process._

object UnoLCMRunner {
    private val logger: Logger = LoggerFactory.getLogger("myLogger")

    def main(args: Array[String]): Unit = {
        var data = new mutable.HashSet[(Int, Set[Int])]
        val lines = Source.fromFile("/home/and/tmp/DB80K_E100.0_M50_Maximals.txt")
        for(line <- lines.getLines()){
            val IdAndTransaction = line.split(";")
            val tid = IdAndTransaction(0).toInt
            val transaction = IdAndTransaction(1).split(" ").map(_.toInt).toList.sorted.toSet
            data += (tid -> transaction)
        }
        lines.close()
        val transactionsByTID = data.groupBy(x => x._1).mapValues(v => v.map(m => m._2))
        for(tid <- transactionsByTID.keys){
            val output = transactionsByTID.get(tid).map(t => t.mkString(" ")).toList.mkString("\n")
            new PrintWriter("/tmp/input%d.txt".format(tid)){
                write(output)
                close()
            }
            logger.info("lcm M /tmp/input%d.txt 1 -".format(tid))
            "lcm M /tmp/input%d.txt 1 -".format(tid) !
        }
    }
}

