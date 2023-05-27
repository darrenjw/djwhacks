/*
window.c

Very simple Gtk app, displaying a window on the console

On Ubuntu, requires the package "libgtk-4-dev"

Compile with:

gcc $(pkg-config --cflags gtk4) -o window window.c $(pkg-config --libs gtk4)


 */


#include <gtk/gtk.h>

static void activate(GtkApplication *app, gpointer user_data) {
  GtkWidget *window;
  GtkWidget *button;
  GtkWidget *image;

  window = gtk_application_window_new(app);
  gtk_window_set_title(GTK_WINDOW(window), "Fern");
  gtk_window_set_default_size(GTK_WINDOW(window), 800, 900);

  image = gtk_image_new_from_file("test8.jpg");
  gtk_window_set_child(GTK_WINDOW(window), image);

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

// eof

