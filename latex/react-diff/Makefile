# Makefile
# On a system with make, latex, pdflatex and beamer, just typing "make" should build the PDF

FILE=srdm

DIAFIGS=1d.pdf

$(FILE).pdf: $(FILE).tex $(DIAFIGS)
	pdflatex $(FILE)
	pdflatex $(FILE)

1d.pdf: 1d.dia
	dia --export 1d.pdf 1d.dia

view: $(FILE).pdf
	acroread $(FILE).pdf &

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
	rm -f *~ core $(FILE).pdf *.ps *.dvi *.log *.aux *.out *.flc

# eof
