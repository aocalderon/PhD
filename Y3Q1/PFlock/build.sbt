name := "PFlock"
organization := "UCR-DBLab"
version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "joda-time" % "joda-time" % "2.9.9"
libraryDependencies += "org.joda" % "joda-convert" % "1.8.1"
libraryDependencies += "org.rogach" % "scallop_2.11" % "2.1.3"
libraryDependencies += "InitialDLab" % "simba_2.11" % "1.0"
libraryDependencies += "org.slf4j" % "slf4j-jdk14" % "1.7.25"
//libraryDependencies += "com.groupon.sparklint" % "sparklint-spark210_2.11" % "1.0.9-SNAPSHOT"

mainClass in (Compile, run) := Some("Runner")
mainClass in (Compile, packageBin) := Some("Runner")
