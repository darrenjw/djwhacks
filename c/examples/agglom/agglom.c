/*
agglom.c

Main runner program

GTK crap

 */


#include <gtk/gtk.h>
#include <cairo.h>

#include <utils.h>


static void draw_cb(GtkDrawingArea *drawing_area, cairo_t *cr, int width, int height, gpointer data) {
  plot_image(cr);
}

int redraw(gpointer data) {
  GtkWidget *drawing_area = data;
  augment_image();
  gtk_widget_queue_draw(drawing_area);
  return(1);
}

static void activate(GtkApplication *app, gpointer user_data) {
  GtkWidget *window;
  GtkWidget *button;
  GtkWidget *drawing_area;
  
  window = gtk_application_window_new(app);
  gtk_window_set_title(GTK_WINDOW(window), "Agglomeration process");
  gtk_window_set_default_size(GTK_WINDOW(window), WIDTH, HEIGHT);
  drawing_area = gtk_drawing_area_new();
  
  gtk_window_set_child(GTK_WINDOW(window), drawing_area);
  gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(drawing_area), draw_cb, NULL, NULL);

  // N.B. May need to lower the refresh rate on slow computers - try changing timeout to 100 microseconds if having problems
  g_timeout_add(10, redraw, drawing_area); // redraw every 10 microseconds (100 Hz)
  
  gtk_window_present(GTK_WINDOW(window));

}


int main(int argc, char **argv) {
  GtkApplication *app;
  int status;

  int i;
  init_image();  // create blank image
  for (i=0;i<100;i++)
    augment_image();
  
  app = gtk_application_new("org.gtk.window", G_APPLICATION_FLAGS_NONE);
  g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
  status = g_application_run(G_APPLICATION(app), argc, argv);
  g_object_unref(app);

  return status;
}



// eof

