import org.osgeo.proj4j._

/**
  * Converts coordinates from EPSG:32632 (WGS 84 / UTM zone 32N) to WGS84,
  * then prints the result to the standard output stream.
  */
object MyApp {
  def main(args: Array[String]): Unit = {
    val csName1 = "EPSG:32636"
    val csName2 = "EPSG:4326"
    val ctFactory = new CoordinateTransformFactory
    val csFactory = new CRSFactory
    /*
         * Create {@link CoordinateReferenceSystem} & CoordinateTransformation.
         * Normally this would be carried out once and reused for all transformations
         */
    val crs1 = csFactory.createFromName(csName1)
    val crs2 = csFactory.createFromName(csName2)
    val trans = ctFactory.createTransform(crs1, crs2)
    /*
         * Create input and output points.
         * These can be constructed once per thread and reused.
         */
    val p1 = new ProjCoordinate
    val p2 = new ProjCoordinate
    p1.x = 500000
    p1.y = 4649776.22482
    System.out.println(p1.toString)
    /*
         * Transform point
         */
    trans.transform(p1, p2)
    System.out.println(p2.toString)
  }
}