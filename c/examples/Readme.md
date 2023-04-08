# Some simple C examples for beginners

In increasing order of complexity:

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


## Some ideas for extra examples

* A simple canvas drawing app, with a bunch of examples (sierpinski triangles, menger sponge, fractal trees, GoL, Langton's ant, etc.), but just writing image to disk
* Using an external library (other than GSL) - eg. an image library?
* Very basic Gtk application - eg. rendering an image in a window?

## Some links

* [gcc](https://gcc.gnu.org/)
* [clang](https://clang.llvm.org/)
* [GNU](https://directory.fsf.org/wiki/GNU)
    * [GNU C library](https://www.gnu.org/software/libc/manual/html_node/index.html)
	    * [Math](https://www.gnu.org/software/libc/manual/html_node/Mathematics.html)
    * [gdb](https://sourceware.org/gdb/current/onlinedocs/gdb.html/)
    * [GNU Scientific library](https://www.gnu.org/software/gsl/doc/html/index.html)

