name := "PFlock"
organization := "UCR-DBLab"
version := "1.0"

scalaVersion := "2.11.8"

//libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.1.0"
//libraryDependencies += "org.apache.spark" % "spark-catalyst_2.11" % "2.1.0"
//libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.1.0"
//libraryDependencies += "com.vividsolutions" % "jts-core" % "1.14.0"
libraryDependencies += "joda-time" % "joda-time" % "2.9.9"
libraryDependencies += "org.rogach" % "scallop_2.11" % "2.1.3"
libraryDependencies += "InitialDLab" % "simba_2.11" % "1.0"

mainClass in (Compile, run) := Some("PFlock")
mainClass in (Compile, packageBin) := Some("PFlock")
