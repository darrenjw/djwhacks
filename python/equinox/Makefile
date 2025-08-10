# Makefile

FORCE:
	make movie.mp4

movie.mp4: pred-adam.pdf
	rm -f movie.mp4
	ffmpeg -framerate 25 -i adam%05d.png movie.mp4

pred-adam.pdf: regress.py
	time python3 regress.py

clean:
	rm -f *~ *.pdf *.png

edit:
	emacs Makefile *.md *.py &

# eof

