# Makefile

FILE=bigdata-talk

$(FILE).pdf: $(FILE).tex
	pdflatex $(FILE)
	pdflatex $(FILE)

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
