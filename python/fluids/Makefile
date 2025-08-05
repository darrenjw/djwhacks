# Makefile


FORCE:
	make tracer.mp4


m00001.png: fluid.py
	time python fluid-jax.py

tracer.mp4: m00001.png
	rm -f tracer.mp4
	ffmpeg -framerate 25 -i m%05d.png tracer.mp4

edit:
	emacs Makefile *.md *.py &

clean:
	rm -f *~ *.pdf *.png tracer.mp4


# eof

