name := "PFlock"
organization := "UCR-DBLab"
version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "joda-time" % "joda-time" % "2.9.9"
libraryDependencies += "org.rogach" % "scallop_2.11" % "2.1.3"
libraryDependencies += "InitialDLab" % "simba_2.11" % "1.0"
//libraryDependencies += "com.groupon.sparklint" % "sparklint-spark210_2.11" % "1.0.9-SNAPSHOT"

mainClass in (Compile, run) := Some("PFlock")
mainClass in (Compile, packageBin) := Some("PFlock")
