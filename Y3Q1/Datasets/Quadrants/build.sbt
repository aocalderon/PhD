name := "DatasetFactory"
organization := "UCR-DBLab"
version := "2.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.wvlet.airframe" %% "airframe-log" % "0.24"

mainClass in (Compile, run) := Some("DatasetFactory")
mainClass in (Compile, packageBin) := Some("DatasetFactory")
