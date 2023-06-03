# Makefile


#CFLAGS=-Wall -Wextra -Wstrict-prototypes -g `pkg-config --cflags gtk4`
CFLAGS=`pkg-config --cflags gtk4`

LIBS=`pkg-config --libs gtk4` -lm

HEADERS=utils.h image.h

OBJECTS=agglom.o utils.o image.o

agglom: $(OBJECTS) $(HEADERS)
	gcc $(OBJECTS) -o agglom $(LIBS)

%.o: %.c $(HEADERS)
	gcc -c -I. $(CFLAGS) $< -o $@ 

run: agglom
	./agglom



edit:
	emacs Makefile *.md *.h *.c &

clean:
	rm -f agglom *.o *~




# eof


