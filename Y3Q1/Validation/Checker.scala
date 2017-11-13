import org.slf4j.{Logger, LoggerFactory}
import org.joda.time.DateTime
import scala.io.Source
import scala.collection.mutable.ListBuffer
import scala.collection.SortedSet
import scala.util.control.Breaks._
import java.io.PrintWriter

object Checker {
    private val logger: Logger = LoggerFactory.getLogger("myLogger")
	
    class Itemset(val line: String) extends Ordered [Itemset]  {
	val items: SortedSet[Int] = line.split(" ").par.map(_.toInt).to[SortedSet]
	val n = items.size
  
	override def toString = "{%s: %d}".format(items.mkString(","), n)

	override def compare(that: Itemset) = {
	    if (this.n > that.n)
		1
	    else if (this.n < that.n)
		-1
	    else
		compareStream(this.items.toStream, that.items.toStream)
	}
  
	def compareStream(x: Stream[Int], y: Stream[Int]): Int = {
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

    def readFile(dataset: String, extension: String = "txt"): Unit = {
	var filename = "%s.%s".format(dataset, extension)
	logger.info("Reading %s".format(filename))
	var itemsets: SortedSet[Itemset] = SortedSet.empty
	val file = Source.fromFile(filename)
	for (line <- file.getLines) {
	    itemsets += new Itemset(line)
	}
	file.close()
	new PrintWriter("%s_sorted.%s".format(dataset, extension)) { 
	    write(itemsets.mkString("\n"))
	    close 
	}
	file.close()
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

    def compareFiles(filename1: String, filename2: String): Unit = {
	val file1 = Source.fromFile(filename1)
	var temp: ListBuffer[Array[Int]] = ListBuffer.empty
	for (line <- file1.getLines)
	    temp += line.substring(1, line.indexOf(":")).split(",").map(_.toInt)
	val itemsets1 = temp.toList
	file1.close()
	val file2 = Source.fromFile(filename2)
	temp = ListBuffer.empty
	for (line <- file2.getLines)
	    temp += line.substring(1, line.indexOf(":")).split(",").map(_.toInt)
	val itemsets2 = temp.toList
	file2.close()
	for(itemset1 <- itemsets1){
	    var found = false
	    for(itemset2 <- itemsets2){
		if(!found && itemset2.intersect(itemset1).sameElements(itemset1)){
		    found = true
		    if(itemset1.size != itemset2.size)
			println("%s has been found in %s ...".format(itemset1.mkString(" "), itemset2.mkString(" ")))
		}
	    }
	    if(!found)
		println("%s has not been found...".format(itemset1.mkString(" ")))
	}
    }
    	
    def main(args: Array[String]): Unit = {
	var extension = "txt"
	var dataset = ""

	dataset = "PFlock_DB20K_PFlock_E40.0_M12"
	Checker.readFile(dataset)

	dataset = "BFE_DB20K_BFE_E40.0_M12"
	Checker.readFile(dataset)

	Checker.compareFiles("PFlock_DB20K_PFlock_E40.0_M12_sorted.txt","BFE_DB20K_BFE_E40.0_M12_sorted.txt")

	//var the_itemset = List(37,450,698,914)
	//Checker.findItemsetInFile("/tmp/BeforeLCM.csv", the_itemset)
	//the_itemset = List(37,450,698,914)
	//Checker.findItemsetInFile("/tmp/AfterLCM.csv", the_itemset)
    }
}
