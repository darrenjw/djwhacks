name := "my-scala-native-test"

scalacOptions ++= Seq("-feature")

enablePlugins(ScalaNativePlugin)
scalaVersion := "2.13.4"
nativeMode := "debug"
nativeGC := "immix"

