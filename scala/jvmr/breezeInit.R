# breezeInit.R
# create a scala interpreter with breeze loaded into the classpath from scratch
# see breeze.R for example usage

breezeInit<-function()
{
  library(jvmr)
  sbtStr="name := \"tmp\"

version := \"0.1\"

libraryDependencies  ++= Seq(
            \"org.scalanlp\" %% \"breeze\" % \"0.10\",
            \"org.scalanlp\" %% \"breeze-natives\" % \"0.10\"
)

resolvers ++= Seq(
            \"Sonatype Snapshots\" at \"https://oss.sonatype.org/content/repositories/snapshots/\",
            \"Sonatype Releases\" at \"https://oss.sonatype.org/content/repositories/releases/\"
)

scalaVersion := \"2.11.2\"

lazy val printClasspath = taskKey[Unit](\"Dump classpath\")

printClasspath := {
  (fullClasspath in Runtime value) foreach {
    e => print(e.data+\"!\")
  }
}
"
  tmps=file(file.path(tempdir(),"build.sbt"),"w")
  cat(sbtStr,file=tmps)
  close(tmps)
  owd=getwd()
  setwd(tempdir())
  cpstr=system2("sbt","printClasspath",stdout=TRUE)
  cpst=cpstr[length(cpstr)]
  cpsp=strsplit(cpst,"!")[[1]]
  cp=cpsp[2:(length(cpsp)-1)]
  #print(cp)
  si=scalaInterpreter(cp,use.jvmr.class.path=FALSE)
  setwd(owd)
  si
}


# eof





