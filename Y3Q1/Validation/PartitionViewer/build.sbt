name := "PartitionViewer"
organization := "UCR-DBLab"
version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.rogach" % "scallop_2.11" % "2.1.3"
libraryDependencies += "InitialDLab" % "simba_2.11" % "1.0"
libraryDependencies += "org.slf4j" % "slf4j-jdk14" % "1.7.25"
resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/releases"
libraryDependencies += "com.storm-enroute" %% "scalameter-core" % "0.8.2"


mainClass in (Compile, run) := Some("PartitionViewer")
mainClass in (Compile, packageBin) := Some("PartitionViewer")
