/*
image.h

Structs and function prototypes for images

 */


typedef struct {
  int r;
  int g;
  int b;
} colour;


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


// eof

