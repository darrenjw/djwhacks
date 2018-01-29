/*
MandelFrame.java

Subclass of JFrame for Mandelbrot set images

*/

import java.awt.*;
import java.awt.image.*;
import java.awt.event.*;
import javax.swing.*;

class MandelFrame extends JFrame {

	protected ImagePanel imagePanel;
	protected BufferedImage bi;
	protected WritableRaster wr;
	protected int xSize,ySize;
	protected double re, im, pix;
	protected int iters;

	public MandelFrame(String title,int xSize,int ySize) {
		super(title);
		this.xSize=xSize; this.ySize=ySize;
		// set up GUI
		bi=new BufferedImage(xSize,ySize,BufferedImage.TYPE_BYTE_GRAY);
		wr=bi.getRaster();
		imagePanel=new ImagePanel(bi);
		this.add(imagePanel);
		this.setSize(xSize,ySize);
		// Set up initial set
		re=-2.5; im=-1.5; pix=0.005; iters=100;
		// Close-window listener
		WindowListener w=new WindowAdapter() {
			public void windowClosing(WindowEvent e) {
				MandelFrame.this.dispose();
				System.exit(0);
				}
			};
		this.addWindowListener(w);
		// Mouse-click listener
		MouseListener m=new MouseAdapter() {
			public void mouseClicked(MouseEvent e) {
       				int x,y;
				x=e.getX(); y=e.getY();
				System.out.println("("+x+","+y+")");
				zoom(x,y);
    			}
		};
		this.addMouseListener(m);
		// Repaint and display
		setMandelbrot();
		repaint();
		setVisible(true);
	}


	public void zoom(int x,int y) {
		re=re+pix*x;
		im=im+pix*y;
		pix=pix/2;
		re=re-pix*xSize/2;
		im=im-pix*ySize/2;
		iters=(int) ((double) iters*1.4);
		System.out.println(""+re+" "+im+" "+pix+" "+iters);
		setMandelbrot();
	}

    public void setMandelbrot() {
	double real,imag,curRe,curIm;
	double scale=255.0/iters;
	curRe=re;
	for (int x=0;x<wr.getWidth();x++) {
		curIm=im;
		for (int y=0;y<wr.getHeight();y++) {
			//System.out.println(""+x+" "+y+" "+re+" "+im);
			real=0;imag=0;
			wr.setSample(x,y,0,0);	
			for (int i=0;i<iters;i++) {
				double newreal=real*real-imag*imag+curRe;
				imag=2*real*imag+curIm;
				real=newreal;
				double abssq=real*real+imag*imag;				
				if (abssq>4) {
					//System.out.println(""+x+" "+y);
					wr.setSample(x,y,0,i*scale);
					break;
				}
			}
			curIm+=pix;
		}
		curRe+=pix;
	    }
	repaint();
	}


}



/* eof */


