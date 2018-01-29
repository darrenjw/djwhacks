/*
Mandelbrot.java


*/

import java.io.*;
import java.awt.image.*;
import javax.swing.*;
import javax.imageio.ImageIO;

class Mandelbrot {


    public static void main(String[] arg)
	throws IOException
    {
	MandelFrame mf=new MandelFrame("My Mandelbrot set",900,700);
	System.out.println("created frame");
	// write the image to disk
	//ImageIO.write(bi,"PNG",new File("test.png"));
    }


}


/* eof */

