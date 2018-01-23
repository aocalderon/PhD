import org.slf4j.{Logger, LoggerFactory}
import org.joda.time.DateTime
import scala.io.Source
import scala.collection.mutable.ListBuffer
import scala.collection.SortedSet
import scala.util.control.Breaks._
import java.io.PrintWriter

object Checker {
  private val logger: Logger = LoggerFactory.getLogger("myLogger")
	
  class Disk(val line: String) extends Ordered [Disk]  {
		val temp = line.split(",")
		val x: Double = temp(0).toDouble
		val y: Double = temp(1).toDouble
		val pids: SortedSet[Long] = temp(2).split(" ").par.map(_.toLong).to[SortedSet]
		val n = pids.size
  
		override def toString = "POINT(%f %f);%s".format(x, y, pids.mkString(" "))

		override def compare(that: Disk) = {
	    if (this.n > that.n)
				1
	    else if (this.n < that.n)
				-1
	    else
				compareStream(this.pids.toStream, that.pids.toStream)
		}
  
	def compareStream(x: Stream[Long], y: Stream[Long]): Int = {
		(x.headOption, y.headOption) match {
			case (Some(xh), Some(yh))  => 
				if (xh == yh) {
					compareStream(x.tail, y.tail)
				} else {
					xh.compare(yh)
				}
			case (Some(_), None) => 1
			case (None, Some(_)) => -1
			case (None, None) => 0
	  }
	}
}

  def saveSortedFile(path: String): String = {
		logger.info("Reading %s".format(path))
    val extension = path.split("\\.").last
    val path_without_extension = path.split("\\.").dropRight(1).mkString(".")
		var disks: SortedSet[Disk] = SortedSet.empty
		val file = Source.fromFile(path)
		for (line <- file.getLines) {
	    disks += new Disk(line)
		}
		file.close()
    val sorted_path = "%s_sorted.%s".format(path_without_extension, extension)
		new PrintWriter(sorted_path) {
	    write(disks.mkString("\n"))
	    close() 
		}
		file.close()
    logger.info("%s has been sorted as %s".format(path, sorted_path))

    sorted_path
  }

  def sortFile(path: String): SortedSet[Disk] = {
    logger.info("Reading %s".format(path))
    val extension = path.split("\\.").last
    val path_without_extension = path.split("\\.").dropRight(1).mkString(".")
    var disks: SortedSet[Disk] = SortedSet.empty
    val file = Source.fromFile(path)
    for (line <- file.getLines) {
      disks += new Disk(line)
    }
    file.close()

    disks
  }

  def findItemsetInFile(filename: String, the_itemset: List[Int]): Unit = {
		logger.info("Reading %s".format(filename))
		var temp: ListBuffer[Array[Int]] = ListBuffer.empty
		val file = Source.fromFile(filename)
		for (line <- file.getLines) {
	    temp += line.split(" ").map(_.toInt)
		}
		val itemsets = temp.toList
		file.close()
		var found = false
		for(itemset <- itemsets){
	    if(!found && itemset.intersect(the_itemset).sameElements(the_itemset)){
				found = true
				println("%s has been found in %s ...".format(the_itemset.mkString(" "), itemset.mkString(" ")))
	    }
		}
  }

  def compareFiles(path1: String, path2: String): Unit = {
		val disks1 = sortFile(path1)
    val disks2 = sortFile(path2)
		for(disk1 <- disks1){
	    var found = false
	    for(disk2 <- disks2){
				if(!found && disk2.pids.intersect(disk1.pids).sameElements(disk1.pids)){
					found = true
					if(disk1.n != disk2.n)
						println("%s has been found in %s ...".format(disk1.toString, disk2.toString))
				}
	    }
	    if(!found)
				println("%s has not been found...".format(disk1.toString))
		}
  }
    	
	def main(args: Array[String]): Unit = {
    val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/and/Documents/PhD/Code/")
    val val_path = "Y3Q1/Validation/"
		var extension = "txt"

    val filename1 = args(0)
    var filename2 = args(1)

    var path1 = "%s%s%s.%s".format(phd_home, val_path, filename1, extension)
    var path2 = "%s%s%s.%s".format(phd_home, val_path, filename2, extension)
		Checker.compareFiles(path1,path2)
  }
}
