# Makefile


FORCE:
	make vx.mp4


vx001.png: fluid.py
	time python fluid.py

vx.mp4: vx001.png
	rm -f vx.mp4
	ffmpeg -framerate 25 -i vx%03d.png vx.mp4

edit:
	emacs Makefile *.md *.py &

clean:
	rm -f *~ *.pdf *.png


# eof

