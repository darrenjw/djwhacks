/*
canvas3.c

Third attempt at a very simple canvas drawing app

Draw Sierpinski triangles

https://en.wikipedia.org/wiki/Sierpi%C5%84ski_triangle

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
void image_tri(image *, int, int, int, int, int, int, colour);

void sierp(image *, int, int, int, int, int, int, int, colour);

int main(int argc, char *argv[]) {
  image *im;
  int i;
  // show construction of Sierpinski triangle
  im = image_alloc(1000, 200);
  for (i=0;i<5;i++) {
    sierp(im, i, 200*i, 180, 200*i + 180, 180, 200*i + 90, 20, red);
  }
  image_write(im, "test3.ppm");
  free(im);
  // a level 7 triangle
  im = image_alloc(1000, 800);
  sierp(im, 7, 100, 750, 900, 750, 500, 50, red);
  image_write(im, "test3a.ppm");
  free(im);
}


void sierp(image * im, int level, int x0, int y0, int x1, int y1,
	   int x2, int y2, colour c) {
  int x01, y01, x12, y12, x02, y02;
  if (level == 0) {
    image_tri(im, x0, y0, x1, y1, x2, y2, c);
  } else {
    x01 = (x0 + x1)/2; y01 = (y0 + y1)/2;
    x12 = (x1 + x2)/2; y12 = (y1 + y2)/2;
    x02 = (x0 + x2)/2; y02 = (y0 + y2)/2;
    sierp(im, level - 1, x0, y0, x01, y01, x02, y02, c);
    sierp(im, level - 1, x1, y1, x01, y01, x12, y12, c);
    sierp(im, level - 1, x2, y2, x02, y02, x12, y12, c);
  }
}


// functions from canvas2.c ...


void image_line(image *im, int x0, int y0, int x1, int y1, colour c) {
  int i, xd, yd, tmp;
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
  if (y1 > y0) {
    for (y=y0;y<y1;y++) {
      x = x0 + (x1-x0)*(y-y0)/(y1-y0);
      xp = x0 + (x2-x0)*(y-y0)/(y2-y0);
      image_line(im, x, y, xp, y, c);
    }
  }
  if (y2 > y1) {
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

