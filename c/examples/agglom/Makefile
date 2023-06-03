# Makefile

CFLAGS=`pkg-config --cflags gtk4`

LIBS=`pkg-config --libs gtk4` -lm

agglom: agglom.o
	gcc -o agglom agglom.o $(LIBS)

%.o: %.c
	gcc -c -o $@ $(CFLAGS) $< 

run: agglom
	./agglom

edit:
	emacs Makefile *.md *.h *.c &

clean:
	rm -f agglom *.o *~

# eof


