/*
ImagePanel.java

Subclass of JPanel for bitmap images

*/

import java.awt.*;
import java.awt.image.*;
import javax.swing.*;

class ImagePanel extends JPanel {

	protected BufferedImage image;

	public ImagePanel(BufferedImage image) {
		this.image=image;
		Dimension dim=new Dimension(image.getWidth(),image.getHeight());
		setPreferredSize(dim);
		setMinimumSize(dim);
		revalidate();
		repaint();
	}

	public void paintComponent(Graphics g) {
		System.out.println("paint ImagePanel");
		g.drawImage(image,0,0,this);
	}



}

/* eof */


