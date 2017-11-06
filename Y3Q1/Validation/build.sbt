name := "Checker"
organization := "UCR-DBLab"
version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies += "joda-time" % "joda-time" % "2.9.9"
libraryDependencies += "org.joda" % "joda-convert" % "1.8.1"
libraryDependencies += "org.slf4j" % "slf4j-jdk14" % "1.7.25"

mainClass in (Compile, run) := Some("Checker")
mainClass in (Compile, packageBin) := Some("Checker")
