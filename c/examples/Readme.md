# Some simple C examples for beginners

In roughly increasing order of complexity:

* [e.c](e.c) - compute e from its series expansion *(for loop)*
* [prime.c](prime.c) - test if a number (provided as a command line arg) is prime *(separate function)*
* [factor.c](factor.c) - factor a number into its prime factors *(recursive function)*
* [sqrt.c](sqrt.c) - square root a number using Newton's method *(while loop)*
* [sin.c](sin.c) - compute the sin of a number (in radians) using trig *(recursive function)*
* [stats.c](stats.c) - compute mean and std-dev of numbers *(reading data from stdin, arrays)*
* [statsp.c](statsp.c) - compute mean and std-dev of numbers *(use pointers rather than explicit array indexing)*
* [statsv.c](statsv.c) - compute mean and std-dev of numbers *(use a typedef'd struct as a vector type)*
* [statsg.c](statsg.c) - compute mean and std-dev of numbers *(using the GSL)*
* [canvas1.c](canvas1.c) - first attempt at a canvas drawing app *(writing a file)*
* [canvas2.c](canvas2.c) - extended canvas drawing app *(lines and filled triangles)*
* [canvas3.c](canvas3.c) - canvas drawing example *(Sierpinski triangles)*
* [canvas4.c](canvas4.c) - extended canvas drawing app *(circles and thick lines)*
* [canvas5.c](canvas5.c) - canvas drawing example *(fractal fern)*
* [canvas6.c](canvas6.c) - drawing the Mandelbrot set *(complex numbers)*
* [canvas7.c](canvas7.c) - drawing the Lorenz attractor *(Euler integration of ODEs)*
* [magick.c](magick.c) - drawing the Mandelbrot set on an ImageMagick image *(MagickWand API)*
* [magickFern.c](magickFern.c) - drawing a fern on an ImageMagick image *(more MagickWand API)*
* [cairo.c](cairo.c) - drawing the Mandelbrot set on a Cairo image *(Cairo canvas API)*
* [window.c](window.c) - simple GTK app to load and display an image *(GUI toolkit)*
* [window2.c](window2.c) - simple GTK app to draw an image using Cairo *(more GUI toolkit)*
* [agglom](agglom/) - simple multi-file GTK application *(multiple files, headers, Makefile)*

## Some links

* [gcc](https://gcc.gnu.org/)
* [clang](https://clang.llvm.org/)
* [GNU](https://directory.fsf.org/wiki/GNU)
    * [GNU C library](https://www.gnu.org/software/libc/manual/html_node/index.html)
	    * [Math](https://www.gnu.org/software/libc/manual/html_node/Mathematics.html)
    * [gdb](https://sourceware.org/gdb/current/onlinedocs/gdb.html/)
    * [make](https://www.gnu.org/software/make/)
    * [GNU Scientific library](https://www.gnu.org/software/gsl/doc/html/index.html)
* [ImageMagick](https://imagemagick.org/)
    * [MagickWand API](https://imagemagick.org/script/magick-wand.php)
* [GTK](https://www.gtk.org/)
    * [Cairo](https://www.cairographics.org/documentation/)
