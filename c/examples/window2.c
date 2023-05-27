/*
window2.c

Very simple Gtk app, creating and image and displaying it on a window

On Ubuntu, requires the package "libgtk-4-dev"

Compile with:

gcc $(pkg-config --cflags gtk4) -o window2 window2.c $(pkg-config --libs gtk4) -lm


 */


#include <gtk/gtk.h>
#include <cairo.h>
#include <complex.h>

void mand(cairo_t *, int, int, double complex, double, int);
int level(double complex, int);


static void draw_cb(GtkDrawingArea *drawing_area, cairo_t *cr, int width, int height, gpointer data) {
  mand(cr, width, height, -2.5 + 1.5*I, 3.0, 60);
}

static void activate(GtkApplication *app, gpointer user_data) {
  GtkWidget *window;
  GtkWidget *button;
  GtkWidget *drawing_area;
  int w, h;
  
  w = 1000; h = 800;

  window = gtk_application_window_new(app);
  gtk_window_set_title(GTK_WINDOW(window), "Mandelbrot set");
  gtk_window_set_default_size(GTK_WINDOW(window), w, h);
  drawing_area = gtk_drawing_area_new();
  
  gtk_window_set_child(GTK_WINDOW(window), drawing_area);
  gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(drawing_area), draw_cb, NULL, NULL);
  
  gtk_window_present(GTK_WINDOW(window));

}


int main(int argc, char **argv) {
  GtkApplication *app;
  int status;
  
  app = gtk_application_new("org.gtk.window", G_APPLICATION_FLAGS_NONE);
  g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
  status = g_application_run(G_APPLICATION(app), argc, argv);
  g_object_unref(app);

  return status;
}



void mand(cairo_t *cr, int w, int h, double complex tl, double i_range, int max_its) {
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

