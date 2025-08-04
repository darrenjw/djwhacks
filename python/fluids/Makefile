# Makefile


FORCE:
	make tracer.mp4


s001.png: fluid.py
	time python fluid-jax.py

tracer.mp4: s001.png
	rm -f vx.mp4
	ffmpeg -framerate 25 -i s%03d.png tracer.mp4

edit:
	emacs Makefile *.md *.py &

clean:
	rm -f *~ *.pdf *.png *.mp4


# eof

