/*
canvas5.c

Fifth attempt at a very simple canvas drawing app

A fractal fern

*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct {
  int r;
  int g;
  int b;
} colour;

colour white = {255, 255, 255};
colour black = {0, 0, 0};
colour red = {255, 0, 0};
colour green = {0, 255, 0};
colour darkGreen = {0, 150, 0};
colour blue = {0, 0, 255};

typedef struct {
  int w;
  int h;
  colour *pixels;
} image;


image * image_alloc(int, int);
void image_free(image *);
colour image_get(image *, int, int);
void image_set(image *, int, int, colour);
void image_write(image *, char *);
void image_line(image *, int, int, int, int, colour);
void image_line_thick(image *, int, int, int, int, double, colour);
void image_tri(image *, int, int, int, int, int, int, colour);
void image_circle(image *, int, int, int, colour);
void image_circle_fill(image *, int, int, int, colour);
void image_quad(image *, int, int, int, int, int, int, int, int, colour);

void fern(image *, int, double, double, double, double, double);

int main(int argc, char *argv[]) {
  image *im;
  int i;
  im = image_alloc(800, 900);
  fern(im, 17, 400, 870, 400, 770, 0.7);
  image_write(im, "test5.ppm");
  free(im);
}


void fern(image * im, int lev, double x0, double y0, double x1, double y1, double squ) {
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
  //image_line(im, round(x0), round(y0), round(x1), round(y1), darkGreen);
  image_line_thick(im, round(x0), round(y0), round(x1), round(y1), th, darkGreen);
  if (lev > 0) {
    xd = x1 - x0;
    yd = y1 - y0;
    // left branch
    xd2 = xd*hs*squ;
    yd2 = yd*hs*squ;
    x2 = x1 + (1.0/sqrt(2))*xd2 + (1.0/sqrt(2))*yd2;
    y2 = y1 - (1.0/sqrt(2))*xd2 + (1.0/sqrt(2))*yd2;
    fern(im, lev - 1, x1, y1, x2, y2, sq*squ);
    // right branch
    xd2 = xd*hs*squ;
    yd2 = yd*hs*squ;
    x2 = x0 + rbf*(x1-x0) + (1.0/sqrt(2))*xd2 - (1.0/sqrt(2))*yd2;
    y2 = y0 + rbf*(y1-y0) + (1.0/sqrt(2))*xd2 + (1.0/sqrt(2))*yd2;
    fern(im, lev - 1, x0 + rbf*(x1-x0), y0 + rbf*(y1-y0), x2, y2, sq*squ);
    // top branch
    xd2 = xd*vs;
    yd2 = yd*vs;
    x2 = x1 + cos(vr)*xd2 - sin(vr)*yd2;
    y2 = y1 + sin(vr)*xd2 + cos(vr)*yd2;
    fern(im, lev - 1, x1, y1, x2, y2, squ);
  }
}



void image_line_thick(image * im, int x0, int y0, int x1, int y1, double th, colour c) {
  double gr, ang;
  int xd, yd;
  if (th <= 1.5) {
    image_line(im, x0, y0, x1, y1, c);
  }
  else {
    if (x1 != x0)
      gr = (y1 - y0)/(x1 - x0);
    else
      gr = INFINITY;
    if (gr != 0) {
      gr = -1.0/gr;
      ang = atan(gr);
    }
    else 
      ang = asin(1.0);
    xd = th*cos(ang)/2.0;
    yd = th*sin(ang)/2.0;
    image_quad(im, x0+xd, y0+yd, x1+xd, y1+yd, x1-xd, y1-yd, x0-xd, y0-yd, c);
  }
}

void image_quad(image * im, int x0, int y0, int x1, int y1, int x2, int y2, int x3, int y3, colour c) {
  // assume coords in cyclic order
  image_tri(im, x0, y0, x1, y1, x2, y2, c);
  image_tri(im, x0, y0, x2, y2, x3, y3, c);
}

void image_circle(image * im, int x, int y, int r, colour c) {
  int i,j;
  for (i=0;i<r/sqrt(2);i++) {
    j = sqrt(r*r - i*i);
    image_set(im, x+j, y+i, c);
    image_set(im, x+j, y-i, c);
    image_set(im, x-j, y+i, c);
    image_set(im, x-j, y-i, c);
    image_set(im, x+i, y+j, c);
    image_set(im, x+i, y-j, c);
    image_set(im, x-i, y+j, c);
    image_set(im, x-i, y-j, c);
  }
}

void image_circle_fill(image * im, int x, int y, int r, colour c) {
  int i,j;
  for (i=0;i<r/sqrt(2);i++) {
    j = sqrt(r*r - i*i);
    image_line(im, x-j, y+i, x+j, y+i, c);
    image_line(im, x-j, y-i, x+j, y-i, c);
    image_line(im, x-i, y+j, x+i, y+j, c);
    image_line(im, x-i, y-j, x+i, y-j, c);
  }
}

void image_line(image *im, int x0, int y0, int x1, int y1, colour c) {
  int i, xd, yd, tmp;
  //printf("line from (%d,%d) to (%d,%d)\n",x0,y0,x1,y1);
  image_set(im, x0, y0, c); // make sure at least one pixel set
  xd = abs(x1 - x0);
  yd = abs(y1 - y0);
  if (xd > yd) { // walk in x direction
    if (x1 < x0) {
      tmp=x0; x0=x1; x1=tmp;
      tmp=y0; y0=y1; y1=tmp;
    }
    if (xd > 0) {
      for (i=x0;i<=x1;i++) {
	tmp = y0 + (i-x0)*(y1-y0)/xd;
	image_set(im, i, tmp, c);
      }
    }
  } else { // walk in y direction
    if (y1 < y0) {
      tmp=x0; x0=x1; x1=tmp;
      tmp=y0; y0=y1; y1=tmp;
    }
    if (yd > 0) {
      for (i=y0;i<=y1;i++) {
	tmp = x0 + (i-y0)*(x1-x0)/yd;
	image_set(im, tmp, i, c);
      }
    }
  }
}

void image_tri(image *im, int x0, int y0, int x1, int y1,
	       int x2, int y2, colour c) {
  int x, xp, y, tmp;
  // manual bubble-sort by y value
  if (y1<y0) {
    tmp=x0; x0=x1; x1=tmp;
    tmp=y0; y0=y1; y1=tmp;
  }
  if (y2<y1) {
    tmp=x1; x1=x2; x2=tmp;
    tmp=y1; y1=y2; y2=tmp;
  }
  if (y1<y0) {
    tmp=x0; x0=x1; x1=tmp;
    tmp=y0; y0=y1; y1=tmp;
  }
  // fill in with horizontal lines
  if (y1>y0) {
    for (y=y0;y<=y1;y++) {
      x = x0 + (x1-x0)*(y-y0)/(y1-y0);
      xp = x0 + (x2-x0)*(y-y0)/(y2-y0);
      image_line(im, x, y, xp, y, c);
    }
  }
  if (y2>y1) {
    for (y=y1;y<=y2;y++) {
      x = x1 + (x2-x1)*(y-y1)/(y2-y1);
      xp = x0 + (x2-x0)*(y-y0)/(y2-y0);
      image_line(im, x, y, xp, y, c);
    }
  }
}

image * image_alloc(int width, int height) {
  image * im;
  int i, j;
  im = malloc(sizeof(image));
  im->w = width;
  im->h = height;
  im->pixels = malloc(width*height*sizeof(colour));
  for (i=0;i<width;i++) {
    for (j=0;j<height;j++) {
      image_set(im, i, j, white);
    }
  }
  return(im);
}

void image_free(image * im) {
  free(im->pixels);
  free(im);
}

colour image_get(image * im, int x, int y) {
  return(im->pixels[y*(im->w) + x]);
}

void image_set(image * im, int x, int y, colour c) {
  //printf("%d %d : %d %d %d\n", x, y, c.r, c.g, c.b);
  if ((x>=0)&(y>=0)&(x < im->w)&(y < im->h))
    im->pixels[y*(im->w) + x] = c;
}

// Output in the plain ASCII "P3" PPM format
// https://netpbm.sourceforge.net/doc/ppm.html
void image_write(image * im, char * fileName) {
  FILE *s;
  int i,j;
  colour c;
  s = fopen(fileName, "w");
  fprintf(s, "P3 %d %d 255\n", im->w, im->h);
  for (j=0;j<(im->h);j++) {
    for (i=0;i<(im->w);i++) {
      c = image_get(im, i, j);
      fprintf(s, "%d %d %d\n", c.r, c.g, c.b);
    }
  }
  fclose(s);
}



// eof

