/*
cairo.c

Using the Cairo image manipulation library:
https://www.cairographics.org/documentation/

On Ubuntu, might require the package "libcairo2-dev"

Compile with:

gcc -o cairo cairo.c `pkg-config --cflags --libs cairo` -lm

 */


#include <stdio.h>
#include <stdlib.h>
#include <cairo.h>
#include <complex.h>

void mand(cairo_t *, cairo_surface_t *, int, int, double complex, double, int);
int level(double complex, int);

int main(int argc, char **argv) {
  cairo_t *cr;
  cairo_surface_t *surface;
  int w, h;
  
  w = 1000; h = 800;
  surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w, h);
  cr = cairo_create(surface);
  
  mand(cr, surface, w, h, -2.5 + 1.5*I, 3.0, 60);
  cairo_surface_write_to_png(surface, "test9.png");

  cairo_destroy(cr);
  cairo_surface_destroy(surface);
}


void mand(cairo_t *cr, cairo_surface_t *surface, int w, int h, double complex tl, double i_range, int max_its) {
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
	cairo_set_source_rgb(cr, 0.0, 0.0, 0.1);
      } else { // not in the set
	cairo_set_source_rgb(cr, (double) lev/max_its, 0.0, 0.0);
      }
      cairo_rectangle(cr, i, j, 1, 1);
      cairo_fill(cr);
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

