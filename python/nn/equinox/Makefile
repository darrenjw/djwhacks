# Makefile

FORCE:
	make clean
	make movie.mp4

movie.mp4: loss.csv
	rm -f movie.mp4
	ffmpeg -framerate 25 -i adam%05d.png movie.mp4

loss.csv: regress.py
	time python3 regress.py

clean:
	rm -f *~ *.pdf *.png movie.mp4 loss.csv

edit:
	emacs Makefile *.md *.py &

# eof

