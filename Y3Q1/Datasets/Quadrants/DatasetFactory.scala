import scala.io.Source
import scala.collection.mutable.ListBuffer
import wvlet.log._

object DatasetFactory extends LogSupport {
    case class Point(id: Long, x: Double, y:Double)

    def main(args: Array[String]): Unit = {
        Logger.setDefaultFormatter(LogFormatter.SourceCodeLogFormatter)
        val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/and/Documents/PhD/Code/")
        val path: String = "Y3Q1/Datasets/Quadrants/"
        val dataset: String = "B20K_SE"
        val extension: String = "csv"

        val filename = s"$phd_home$path$dataset.$extension"
        val SE = Source.fromFile(filename)

        var points = new ListBuffer[Point]
        var n = 0
        for(line <- SE.getLines){
            val splittedLine = line.split(",")
            val x = splittedLine(1).toDouble
            val y = splittedLine(2).toDouble
            points += new Point(n, x, y)
            n = n + 1
        }
			
        var min_x: Double = Double.MaxValue
        var min_y: Double = Double.MaxValue
        var max_x: Double = Double.MinValue
        var max_y: Double = Double.MinValue

        points.foreach{ point =>
            val x = point.x
            val y = point.y
            if(x < min_x){
                min_x = x
            }
            if(y < min_y){
                min_y = y
            }
            if(x > max_x){
                max_x = x
            }
            if(y > max_y){
                max_y = y
            }    
        }
        info("(%.2f, %.2f) (%.2f, %.2f)".format(min_x, min_y, max_x, max_y))
	}
}
