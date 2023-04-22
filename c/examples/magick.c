/*
magick.c

Using the MagickWand image manipulation library:
https://imagemagick.org/script/magick-wand.php

On Ubuntu, install the package "libmagickwand-dev"

Compile with:

gcc -o magick magick.c `pkg-config --cflags --libs MagickWand` -lm


 */


#include <stdio.h>
#include <stdlib.h>
#include <wand/MagickWand.h>
#include <complex.h>

void mand(PixelWand *, DrawingWand *, int, int, double complex, double, int);
int level(double complex, int);

int main(int argc, char **argv) {
  MagickWand *m_wand = NULL;
  DrawingWand *d_wand = NULL;
  PixelWand *c_wand = NULL;
  int w, h;
  
  w = 1000; h = 800;
  MagickWandGenesis();
  m_wand = NewMagickWand();
  d_wand = NewDrawingWand();
  c_wand = NewPixelWand();
  
  PixelSetColor(c_wand, "white");
  MagickNewImage(m_wand, w, h, c_wand);
  mand(c_wand, d_wand, w, h, -2.5 + 1.5*I, 3.0, 60);
  MagickDrawImage(m_wand, d_wand);
  MagickWriteImage(m_wand, "test7.jpg");

  c_wand = DestroyPixelWand(c_wand);
  m_wand = DestroyMagickWand(m_wand);
  d_wand = DestroyDrawingWand(d_wand);
  MagickWandTerminus();
}


void mand(PixelWand * c_wand, DrawingWand * d_wand, int w, int h, double complex tl, double i_range, int max_its) {
  int i, j;
  double complex c;
  double res;
  int lev;
  char col_str[80];
  res = i_range/h;
  for (i=0;i<w;i++) {
    for (j=0;j<h;j++) {
      c = tl + i*res - j*res*I;
      lev = level(c, max_its);
      if (lev == -1) { // in the set
	PixelSetColor(c_wand, "rgb(0,0,30)");
      } else { // not in the set
	sprintf(col_str, "rgb(%d,0,0)", 255*lev/max_its);
	//printf("%s\n", col_str);
	PixelSetColor(c_wand, col_str);
      }
      DrawSetFillColor(d_wand, c_wand);
      DrawPoint(d_wand, i, j);
    }
  }
}

int level(double complex c, int max_its) {
  double complex z;
  int i;
  z = 0.0 + 0.0*I;
  for (i=0;i<max_its;i++) {
    z = z*z + c;
    if (cabs(z) > 2.0) // not in the set
      return(i);
  }
  return(-1);
}


// eof

