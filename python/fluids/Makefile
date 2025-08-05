# Makefile


FORCE:
	make tracer.mp4


m001.png: fluid.py
	time python fluid-jax.py

tracer.mp4: m001.png
	rm -f tracer.mp4
	ffmpeg -framerate 25 -i m%03d.png tracer.mp4

edit:
	emacs Makefile *.md *.py &

clean:
	rm -f *~ *.pdf *.png


# eof

