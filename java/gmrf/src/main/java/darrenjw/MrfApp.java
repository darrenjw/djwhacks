import java.io.*;

class MrfApp {

    public static void main(String[] arg)
    throws IOException
    {
    Mrf mrf;
    System.out.println("started program");
    mrf=new Mrf(800,600);
    System.out.println("created mrf object");
    mrf.update(1000);
    System.out.println("done updates");
    mrf.saveImage("mrf.png");
    System.out.println("finished program");
    mrf.frame.dispose();
    System.exit(0);
    }

}


