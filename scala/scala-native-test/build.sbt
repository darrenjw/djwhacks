name := "my-scala-native-test"

scalacOptions ++= Seq("-feature")

enablePlugins(ScalaNativePlugin)
scalaVersion := "2.11.8"
nativeMode := "debug"
nativeGC := "immix"

