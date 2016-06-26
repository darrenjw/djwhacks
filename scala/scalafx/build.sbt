// Name of the project
name := "Hello SBT"
 
// Project version
version := "1.0"
 
// Version of Scala used by the project
scalaVersion := "2.11.7"
 
// Add dependency on ScalaFX library
libraryDependencies += "org.scalafx" %% "scalafx" % "8.0.92-R10"

// Add dependency on JavaFX library based on JAVA_HOME variable
unmanagedJars in Compile += Attributed.blank(file(System.getenv("JAVA_HOME") + "/jre/lib/jfxrt.jar"))

