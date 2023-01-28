# Makefile

UTILPATH=/home/darren/src/git/logreg/Dex

LIBPATH=BUILTIN_LIBRARIES:$(UTILPATH):.

FORCE:
	make lv.html

%.tsv: %.dx
	time dex --lib-path $(LIBPATH) script $<

%.html: %.dx
	time dex --lib-path $(LIBPATH) script $< --outfmt html > $@

.PRECIOUS: %.html
%: %.html
	make $<
	xdg-open $<

edit:
	emacs Makefile *.md *.R *.py *.dx &

clean:
	rm -f *.out *.tsv *.html *~ *.pdf

repl:
	dex --lib-path $(LIBPATH) repl

sandpit:
	dex --lib-path $(LIBPATH) script sandpit.dx

random:
	dex --lib-path $(LIBPATH) script random-tests.dx


# eof
