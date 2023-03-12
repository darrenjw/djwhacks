# Makefile




FORCE:
	sbt run


edit:
	emacs Makefile *.sbt *.R src/main/scala/*.scala &

todo:
	grep TODO: src/main/scala/*.scala


clean:
	rm -f *~ mrf-????.png mrf.csv movie.mp4 movie-a.mp4 *.pdf
	sbt clean
	rm -rf target


# eof


