name := "hello"

version := "1.0"

libraryDependencies  ++= Seq(
            // other dependencies here
            // pick and choose:
            "org.scalanlp" % "breeze-math_2.9.1" % "0.2.1",
            "org.scalanlp" % "breeze-viz_2.9.1" % "0.2.1"
)

resolvers ++= Seq(
            // other resolvers here
            "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/releases/"
)


