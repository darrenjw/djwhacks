# Makefile

UTILPATH=/home/ndjw1/src/git/logreg/Dex

LIBPATH=BUILTIN_LIBRARIES:$(UTILPATH):.

FORCE:
	make lv.tsv

%.tsv: %.dx
	time dex --lib-path $(LIBPATH) script $<

%.out: %.tsv
	./analyse.R $< > $@
	mv Rplots.pdf $@.pdf

edit:
	emacs Makefile *.md *.R *.py *.dx &

clean:
	rm -f *.out *.tsv *~ *.pdf

repl:
	dex --lib-path $(LIBPATH) repl

sandpit:
	dex --lib-path $(LIBPATH) script sandpit.dx

random:
	dex --lib-path $(LIBPATH) script random-tests.dx


# eof
