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


To use the linear algebra stuff, need to start sbt with:

sbt -J"-Dcom.github.fommil.netlib.BLAS=com.github.fommiletlib.NativeRefBLAS -Dcom.github.fommil.netlib.LAPACK=com.github.fommil.netlib.NativeRefLAPACK"

"That's probably a bug against netlib-java. Can you take a look at the 
readme here: https://github.com/fommil/netlib-java and see if any of 
the suggestions work? One that should almost certainly work is to 
specify 
-Dcom.github.fommil.netlib.BLAS=com.github.fommil.netlib.F2jBLAS 
-Dcom.github.fommil.netlib.LAPACK=com.github.fommil.netlib.F2jLAPACK 
-Dcom.github.fommil.netlib.ARPACK=com.github.fommil.netlib.F2jARPACK 
as JVM parameters "

Things will be slower, but they'll work. 


To work in the ScalaIDE (Eclipse), first make sure that the "sbteclipse" plugin for sbt is installed (config file in ~/.sbt), and then use the "eclipse" sbt target to produce an eclipse project folder so that the project can be imported into Eclipse as an existing project with all the dependencies set up correctly...



