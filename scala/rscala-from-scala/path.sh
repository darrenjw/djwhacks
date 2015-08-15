RSCALA_JAR=$(R --slave -e 'library("rscala"); cat(rscala::rscalaJar("2.11"))')
echo $RSCALA_JAR

