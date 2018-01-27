import java.io.*;
import java.util.*;
import java.awt.image.*;
import javax.swing.*;
import javax.imageio.ImageIO;
import cern.jet.random.tdouble.*;
import cern.jet.random.tdouble.engine.*;
import cern.colt.matrix.tdouble.impl.*;
 
class Mrf 
{
    int n,m;
    DenseDoubleMatrix2D cells;
    DoubleRandomEngine rng;
    Normal rngN;
    BufferedImage bi;
    WritableRaster wr;
    JFrame frame;
    ImagePanel ip;
     
    Mrf(int n_arg,int m_arg,DoubleRandomEngine rng)
    {
    n=n_arg;
    m=m_arg;
    cells=new DenseDoubleMatrix2D(n,m);
    this.rng=rng;
    rngN=new Normal(0.0,1.0,rng);
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
        if (cells.getQuick(i,j)>mx) { mx=cells.getQuick(i,j); }
        if (cells.getQuick(i,j)<mn) { mn=cells.getQuick(i,j); }
        }
    }
    for (int i=0;i<n;i++) {
        for (int j=0;j<m;j++) {
        int level=(int) (255*(cells.getQuick(i,j)-mn)/(mx-mn));
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
            mean=0.5*(cells.getQuick(0,1)+cells.getQuick(1,0));
            } 
            else if (j==m-1) {
            mean=0.5*(cells.getQuick(0,j-1)+cells.getQuick(1,j));
            } 
            else {
            mean=(cells.getQuick(0,j-1)+cells.getQuick(0,j+1)+cells.getQuick(1,j))/3.0;
            }
        }
        else if (i==n-1) {
            if (j==0) {
            mean=0.5*(cells.getQuick(i,1)+cells.getQuick(i-1,0));
            }
            else if (j==m-1) {
            mean=0.5*(cells.getQuick(i,j-1)+cells.getQuick(i-1,j));
            }
            else {
            mean=(cells.getQuick(i,j-1)+cells.getQuick(i,j+1)+cells.getQuick(i-1,j))/3.0;
            }
        }
        else if (j==0) {
            mean=(cells.getQuick(i-1,0)+cells.getQuick(i+1,0)+cells.getQuick(i,1))/3.0;
        }
        else if (j==m-1) {
            mean=(cells.getQuick(i-1,j)+cells.getQuick(i+1,j)+cells.getQuick(i,j-1))/3.0;
        }
        else {
            mean=0.25*(cells.getQuick(i,j-1)+cells.getQuick(i,j+1)+cells.getQuick(i+1,j)
                   +cells.getQuick(i-1,j));
        }
        cells.setQuick(i,j,mean+rngN.nextDouble());
        }
    }
    updateImage();
    }
     
}


