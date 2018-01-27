
import java.io.*;
import cern.jet.random.tdouble.engine.*;
 
class MrfApp {
 
    public static void main(String[] arg)
    throws IOException
    {
    Mrf mrf;
    int seed=1234;
    System.out.println("started program");
        DoubleRandomEngine rngEngine=new DoubleMersenneTwister(seed);
    mrf=new Mrf(800,600,rngEngine);
    System.out.println("created mrf object");
    mrf.update(1000);
    System.out.println("done updates");
    mrf.saveImage("mrf.png");
    System.out.println("finished program");
    mrf.frame.dispose();
    System.exit(0);
    }
 
}

