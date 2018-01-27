import java.io.*;
import java.util.*;
import java.awt.image.*;
import javax.swing.*;
import javax.imageio.ImageIO;
 
 
class Mrf 
{
    int n,m;
    double[][] cells;
    Random rng;
    BufferedImage bi;
    WritableRaster wr;
    JFrame frame;
    ImagePanel ip;
     
    Mrf(int n_arg,int m_arg)
    {
    n=n_arg;
    m=m_arg;
    cells=new double[n][m];
    rng=new Random();
    bi=new BufferedImage(n,m,BufferedImage.TYPE_BYTE_GRAY);
    wr=bi.getRaster();
    frame=new JFrame("MRF");
    frame.setSize(n,m);
    frame.add(new ImagePanel(bi));
    frame.setVisible(true);
    }
     
    public void saveImage(String filename)
    throws IOException
    {
    ImageIO.write(bi,"PNG",new File(filename));
    }
     
    public void updateImage()
    {
    double mx=-1e+100;
    double mn=1e+100;
    for (int i=0;i<n;i++) {
        for (int j=0;j<m;j++) {
        if (cells[i][j]>mx) { mx=cells[i][j]; }
        if (cells[i][j]<mn) { mn=cells[i][j]; }
        }
    }
    for (int i=0;i<n;i++) {
        for (int j=0;j<m;j++) {
        int level=(int) (255*(cells[i][j]-mn)/(mx-mn));
        wr.setSample(i,j,0,level);
        }
    }
    frame.repaint();
    }
     
    public void update(int num)
    {
    for (int i=0;i<num;i++) {
        updateOnce();
    }
    }
     
    private void updateOnce()
    {
    double mean;
    for (int i=0;i<n;i++) {
        for (int j=0;j<m;j++) {
        if (i==0) {
            if (j==0) {
            mean=0.5*(cells[0][1]+cells[1][0]);
            } 
            else if (j==m-1) {
            mean=0.5*(cells[0][j-1]+cells[1][j]);
            } 
            else {
            mean=(cells[0][j-1]+cells[0][j+1]+cells[1][j])/3.0;
            }
        }
        else if (i==n-1) {
            if (j==0) {
            mean=0.5*(cells[i][1]+cells[i-1][0]);
            }
            else if (j==m-1) {
            mean=0.5*(cells[i][j-1]+cells[i-1][j]);
            }
            else {
            mean=(cells[i][j-1]+cells[i][j+1]+cells[i-1][j])/3.0;
            }
        }
        else if (j==0) {
            mean=(cells[i-1][0]+cells[i+1][0]+cells[i][1])/3.0;
        }
        else if (j==m-1) {
            mean=(cells[i-1][j]+cells[i+1][j]+cells[i][j-1])/3.0;
        }
        else {
            mean=0.25*(cells[i][j-1]+cells[i][j+1]+cells[i+1][j]
                   +cells[i-1][j]);
        }
        cells[i][j]=mean+rng.nextGaussian();
        }
    }
    updateImage();
    }
     
}


