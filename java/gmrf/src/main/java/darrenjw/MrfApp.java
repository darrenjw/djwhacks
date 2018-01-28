import java.io.*;

class MrfApp {

    public static void main(String[] arg)
    throws IOException
    {
    Mrf mrf;
    int n = 1000;
    int w = 800;
    int h = 600;
    System.out.println("started program");
    System.out.println(String.format("Simulating %d iterations on a %dx%d grid",n,w,h));
    mrf=new Mrf(w,h);
    System.out.println("created mrf object");
    mrf.update(n);
    System.out.println("done updates");
    mrf.saveImage("mrf.png");
    System.out.println("finished program");
    mrf.frame.dispose();
    System.exit(0);
    }

}


