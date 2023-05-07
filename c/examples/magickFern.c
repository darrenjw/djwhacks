/*
magickFern.c

Using the MagickWand image manipulation library:
https://imagemagick.org/script/magick-wand.php

On Ubuntu, install the package "libmagickwand-dev"

Compile with:

gcc -o magickFern magickFern.c `pkg-config --cflags --libs MagickWand` -lm


 */

#include <stdio.h>
#include <stdlib.h>
#include <wand/MagickWand.h>

void fern(DrawingWand *, int, double, double, double, double, double);


int main(int argc, char **argv) {
  MagickWand *m_wand = NULL;
  DrawingWand *d_wand = NULL;
  PixelWand *c_wand = NULL;
  int w, h;
  
  w = 800; h = 900;
  MagickWandGenesis();
  m_wand = NewMagickWand();
  d_wand = NewDrawingWand();
  c_wand = NewPixelWand();
  
  PixelSetColor(c_wand, "white");
  MagickNewImage(m_wand, w, h, c_wand);
  PixelSetColor(c_wand, "darkGreen");
  DrawSetStrokeColor(d_wand, c_wand);
  DrawSetStrokeAntialias(d_wand, 1);
  fern(d_wand, 15, 400, 870, 400, 770, 0.7);
  MagickDrawImage(m_wand, d_wand);
  MagickWriteImage(m_wand, "test8.jpg");

  c_wand = DestroyPixelWand(c_wand);
  m_wand = DestroyMagickWand(m_wand);
  d_wand = DestroyDrawingWand(d_wand);
  MagickWandTerminus();
}


void fern(DrawingWand * d_wand, int lev, double x0, double y0, double x1, double y1, double squ) {
  double th, l, xd, yd, xd2, yd2, x2, y2, tc, vs, hs, sq, vr, rbf;
  //printf("%d\n", lev);
  tc = 0.05; // thickness coef
  hs = 0.6; // horizontal shrink factor
  sq = 0.7; // horizontal squish factor
  vs = 0.9; // vertical shrink factor
  rbf = 0.7; // right branch fraction
  vr = 0.03; // vertical rotation angle (radians)
  l = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
  th = tc*l;
  DrawSetStrokeWidth(d_wand, th);
  DrawLine(d_wand, round(x0), round(y0), round(x1), round(y1));
  if (lev > 0) {
    xd = x1 - x0;
    yd = y1 - y0;
    // left branch
    xd2 = xd*hs*squ;
    yd2 = yd*hs*squ;
    x2 = x1 + (1.0/sqrt(2))*xd2 + (1.0/sqrt(2))*yd2;
    y2 = y1 - (1.0/sqrt(2))*xd2 + (1.0/sqrt(2))*yd2;
    fern(d_wand, lev - 1, x1, y1, x2, y2, sq*squ);
    // right branch
    xd2 = xd*hs*squ;
    yd2 = yd*hs*squ;
    x2 = x0 + rbf*(x1-x0) + (1.0/sqrt(2))*xd2 - (1.0/sqrt(2))*yd2;
    y2 = y0 + rbf*(y1-y0) + (1.0/sqrt(2))*xd2 + (1.0/sqrt(2))*yd2;
    fern(d_wand, lev - 1, x0 + rbf*(x1-x0), y0 + rbf*(y1-y0), x2, y2, sq*squ);
    // top branch
    xd2 = xd*vs;
    yd2 = yd*vs;
    x2 = x1 + cos(vr)*xd2 - sin(vr)*yd2;
    y2 = y1 + sin(vr)*xd2 + cos(vr)*yd2;
    fern(d_wand, lev - 1, x1, y1, x2, y2, squ);
  }
}



// eof

