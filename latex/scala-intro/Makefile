# Makefile
# On a system with make, latex, pdflatex and beamer, just typing "make" should build the PDF

FILE=scala-talk

$(FILE).pdf: $(FILE).tex
	pdflatex $(FILE)
	pdflatex $(FILE)

$(FILE).ps: $(FILE).dvi
	dvips $(FILE)

$(FILE).dvi: $(FILE).tex
	latex $(FILE)
	latex $(FILE)

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
	rm -f *~ core *.pdf *.ps *.dvi *.log *.aux *.out *.flc

# eof
