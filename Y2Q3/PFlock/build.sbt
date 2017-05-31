name := "PFlock"
organization := "UCR-DBLab"
version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.1.0"
libraryDependencies += "org.apache.spark" % "spark-catalyst_2.11" % "2.1.0"
libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.1.0"

libraryDependencies += "com.vividsolutions" % "jts-core" % "1.14.0"

mainClass in (Compile, run) := Some("main.scala.PFlock")
mainClass in (Compile, packageBin) := Some("main.scala.PFlock")