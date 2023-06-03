/*
agglom.c

Main runner program

 */


#include <gtk/gtk.h>
#include <cairo.h>

#include <utils.h>


static void draw_cb(GtkDrawingArea *drawing_area, cairo_t *cr, int width, int height, gpointer data) {
  plot_image(cr);
}

void redraw(GtkDrawingArea *drawing_area) {
  augment_image();
  gtk_widget_queue_draw(drawing_area);
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

  g_timeout_add(1, redraw, drawing_area);
  
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

