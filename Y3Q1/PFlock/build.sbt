name := "PFlock"
organization := "UCR-DBLab"
version := "2.0"

scalaVersion := "2.11.8"

libraryDependencies += "joda-time" % "joda-time" % "2.9.9"
libraryDependencies += "org.joda" % "joda-convert" % "1.8.1"
libraryDependencies += "org.rogach" % "scallop_2.11" % "2.1.3"
libraryDependencies += "InitialDLab" % "simba_2.11" % "1.0"
libraryDependencies += "org.slf4j" % "slf4j-jdk14" % "1.7.25"
libraryDependencies += "com.github.filosganga" % "geogson-core" % "1.2.21"

//libraryDependencies += "fr.liglab.jlcm" % "jLCM" % "1.7.0" 
//libraryDependencies += "org.wvlet.airframe" %% "airframe-log" % "0.24"
//libraryDependencies += "com.groupon.sparklint" % "sparklint-spark210_2.11" % "1.0.9-SNAPSHOT"

mainClass in (Compile, run) := Some("FlockFinder")
mainClass in (Compile, packageBin) := Some("FlockFinder")
