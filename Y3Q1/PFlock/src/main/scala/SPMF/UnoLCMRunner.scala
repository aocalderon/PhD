package SPMF

import org.slf4j.{Logger, LoggerFactory}

import scala.language.postfixOps
import sys.process._

object UnoLCMRunner {
    private val logger: Logger = LoggerFactory.getLogger("myLogger")

    def main(args: Array[String]): Unit = {
        logger.info("Running LCM...")
        "lcm M /home/acald013/PhD/Y3Q1/PFlock/src/main/scala/SPMF/input2.txt 1 -" !
    }
}

