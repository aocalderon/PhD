import scala.io.Source
import scala.collection.mutable.ListBuffer
import wvlet.log._

object DatasetFactory extends LogSupport {
    case class Point(id: Long, x: Double, y:Double)

    def main(args: Array[String]): Unit = {
        Logger.setDefaultFormatter(LogFormatter.SourceCodeLogFormatter)
        val delta: Double = 1000
        val phd_home = scala.util.Properties.envOrElse("PHD_HOME", "/home/and/Documents/PhD/Code/")
        val path: String = "Y3Q1/Datasets/Original/"
        val dataset: String = "B60K"
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
		new java.io.PrintWriter(s"$phd_home$path${dataset}_0.$extension") {
			write(points.toList.map(p => "%d,%f,%f".format(p.id, p.x, p.y)).mkString("\n"))
			close()
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
        val extent_y = max_y - min_y
        val extent_x = max_x - min_x
        info("(%.2f, %.2f)".format(extent_x, extent_y))

        var above = new ListBuffer[Point]
        points.zipWithIndex
			.foreach{ case(point, index) =>
				above += new Point(index + n, point.x, point.y + extent_y + delta)
			}
		new java.io.PrintWriter(s"$phd_home$path${dataset}_1.$extension") {
			write(above.toList.map(p => "%d,%f,%f".format(p.id, p.x, p.y)).mkString("\n"))
			close()
		}

        var below = new ListBuffer[Point]
        points.zipWithIndex
			.foreach{ case(point, index) =>
				below += new Point(index + (2 * n), point.x, point.y - extent_y - delta)
			}
		new java.io.PrintWriter(s"$phd_home$path${dataset}_2.$extension") {
			write(below.toList.map(p => "%d,%f,%f".format(p.id, p.x, p.y)).mkString("\n"))
			close()
		}

        var below2 = new ListBuffer[Point]
        below.zipWithIndex
			.foreach{ case(point, index) =>
				below2 += new Point(index + (3 * n), point.x, point.y - extent_y - delta)
			}
		new java.io.PrintWriter(s"$phd_home$path${dataset}_3.$extension") {
			write(below2.toList.map(p => "%d,%f,%f".format(p.id, p.x, p.y)).mkString("\n"))
			close()
		}
	}
}
