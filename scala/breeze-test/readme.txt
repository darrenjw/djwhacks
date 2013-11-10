Breeze test

My usual Gibbs sampler script re-written to use the Breeze library

Designed to be used from "sbt".

Should just "run" from sbt.

For timing purposes, can run from the command line (after building with sbt using "package" task) using a command like:

time scala -cp /var/tmp/commons-math3-3.0/commons-math3-3.0.jar:/var/tmp/breeze-math_2.10-0.4.jar target/scala-2.10/breeze-test_2.10-0.1.jar > data.txt

Edit classpath as needed... eg., 

time scala -cp /var/tmp/commons-math3-3.0/commons-math3-3.0.jar:/home/ndjw1/src/git/breeze/target/scala-2.10/breeze_2.10-0.6-SNAPSHOT.jar target/scala-2.10/breeze-test_2.10-0.1.jar > data.txt

to use my home-built snapshot... 
For the git clone of breeze, run
git pull
sbt
compile
package
to get the new snapshot jar
