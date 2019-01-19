# Makefile
# On a system with make, latex, pdflatex and beamer, just typing "make" should build the PDF

FILE=talk-robots


$(FILE).pdf: $(FILE).tex
	texi2pdf $(FILE).tex

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
	rm -f *~ core $(FILE).pdf *.ps *.eps *.dvi *.log *.aux *.out *.flc

# eof
