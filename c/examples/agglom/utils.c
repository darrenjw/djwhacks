/*
utils.c

Utility functions required to generate the agglomeration process

 */


#include <gtk/gtk.h>
#include <cairo.h>

#include <utils.h>
#include <image.h>

image * im;

// copy the image to the GTK window
void plot_image(cairo_t *cr) {
  int x, y;
  colour col;
  for (x=0;x<WIDTH;x++) {
    for (y=0;y<HEIGHT;y++) {
      col=image_get(im, x, y);
      cairo_set_source_rgb(cr, (float) col.r/255,
			   (float) col.g/255, (float) col.b/255);
      cairo_rectangle(cr, x, y, 1, 1);
      cairo_fill(cr);
    }
  }
}

// add one more pixel to the image
void augment_image() {
  int x, y, dir;
  colour col = {0, 120, 40};
  x = rand() % WIDTH;
  // select diffusion initialisation strategy below
  //y = 0; // initialise with particle random on top row
  y = rand() % HEIGHT; // initialise at a random point on image
  while (!is_adjacent(x,y)) {
    dir = rand() % 4;
    if (dir == 0)
      y--;
    else if (dir == 1)
      x--;
    else if (dir == 2)
      y++;
    else if (dir == 3)
      x++;
    if (x<0)
      x = 0;
    if (y<0)
      y = 0;
    if (x >= WIDTH)
      x = WIDTH - 1;
    if (y >= HEIGHT)
      y = HEIGHT - 1;
  }
  image_set(im, x, y, col);
}

// check if this location is adjacent to an occupied pixel
int is_adjacent(int x, int y) {
  int adj = 0;
  // up
  if (y>0)
    if (is_occupied(x, y-1))
      adj=1;
  // left
  if (x>0)
    if (is_occupied(x-1, y))
      adj=1;
  // down
  if (y < HEIGHT-1)
    if (is_occupied(x, y+1))
      adj=1;
  // right
  if (x < WIDTH-1)
    if (is_occupied(x+1, y))
      adj=1;
  return adj;
}

// check if this pixel is occupied
int is_occupied(int x, int y) {
  colour col;
  col = image_get(im, x, y);
  if ((col.r < 255)|(col.g < 255)|(col.b < 255))
    return 1;
  else
    return 0;
}

// initialise the image (with a line along half the bottom)
void init_image() {
  int i;
  colour col = {0, 0, 0};
  srand(time(NULL)); // seed the (crumby) random number generator
  im = image_alloc(WIDTH, HEIGHT);
  for (i=WIDTH/4; i<3*WIDTH/4; i++)
    image_set(im, i, HEIGHT-1, col);
}


// eof

