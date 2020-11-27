# Makefile

FILE=djw-streaming

FORCE:
	make $(FILE).pdf

view: $(FILE).pdf
	xdg-open $(FILE).pdf &

present: $(FILE).pdf
	okular $(FILE).pdf &

$(FILE).pdf: $(FILE).tex content.tex
	#cd figs; make
	pdflatex $(FILE)
	pdflatex $(FILE)

$(FILE)-nup.pdf: $(FILE).pdf
	pdfnup --nup 3x3 $(FILE).pdf

content.tex: content.md
	pandoc -t beamer -Vtheme:metropolis content.md -o content.tex

clean:
	rm -f *~ $(FILE).pdf content.tex *.nav *.aux

edit:
	emacs Makefile $(FILE).tex *.txt *.md &


# eof
