/*
utils.h

Function prototypes for functions in utils.c

*/

#include <gtk/gtk.h>
#include <cairo.h>

// some useful "global" definitions
#define WIDTH 800
#define HEIGHT 600

// functions defined in utils.c
void plot_image(cairo_t *);
void init_image(void);
void augment_image(void);
int is_occupied(int, int);
int is_adjacent(int, int);




// eof

  

  
