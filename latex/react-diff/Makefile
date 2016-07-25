# Makefile
# On a system with make, latex, pdflatex and beamer, just typing "make" should build the PDF

FILE=srdm

DIAFIGS=1d.pdf 2d.pdf

$(FILE).pdf: $(FILE).tex $(DIAFIGS)
	pdflatex $(FILE)
	pdflatex $(FILE)

1d.pdf: 1d.dia
	dia --export=1d.eps 1d.dia
	epstopdf 1d.eps

2d.pdf: 2d.dia
	dia --export=2d.eps 2d.dia
	epstopdf 2d.eps

view: $(FILE).pdf
	xdg-open $(FILE).pdf &

present: $(FILE).pdf
	okular $(FILE).pdf

edit:
	gedit $(FILE).tex &

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
