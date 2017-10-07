import org.rogach.scallop.{ScallopConf, ScallopOption}

object Tester {
  class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val estart: ScallopOption[Double] = opt[Double](default = Some(10.0))
    val estep:  ScallopOption[Double] = opt[Double](default = Some(10.0))
    val eend:   ScallopOption[Double] = opt[Double](default = Some(20.0))
    val mstart: ScallopOption[Int] = opt[Int](default = Some(4))
    val mstep:  ScallopOption[Int] = opt[Int](default = Some(2))
    val mend:   ScallopOption[Int] = opt[Int](default = Some(4))
    verify()
  }
    
  def main(args: Array[String]): Unit = {
    val conf = new Conf(args)
    for( epsilon <- conf.estart() to conf.eend() by conf.estep();
         mu <- conf.mstart() to conf.mend() by conf.mstep()){
      FlockFinder
    }
  }
}
