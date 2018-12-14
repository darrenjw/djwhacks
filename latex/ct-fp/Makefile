# Makefile
# On a system with make, latex, pdflatex and beamer, just typing "make" should build the PDF

FILE=ctfp

DIAFIGS=

$(FILE).pdf: $(FILE).tex $(DIAFIGS)
	texi2pdf $(FILE).tex

$(FILE)-nup.pdf: $(FILE).pdf
	pdfnup --nup 4x4 $(FILE).pdf

1d.pdf: 1d.dia
	dia --export=1d.eps 1d.dia
	epstopdf 1d.eps

view: $(FILE).pdf
	xdg-open $(FILE).pdf &

present: $(FILE).pdf
	pdf-presenter-console $(FILE).pdf

edit:
	emacs $(FILE).tex &

update:
	git pull
	git log | less

commit:
	git commit -a
	git push
	make update

clean:
	rm -f *~ core $(FILE).pdf $(DIAFIGS) *.ps *.eps *.dvi *.log *.aux *.out *.flc

# eof
