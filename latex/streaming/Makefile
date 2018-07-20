# Makefile

FILE=djw-streaming

FORCE:
	make $(FILE).pdf

view: $(FILE).pdf
	xdg-open $(FILE).pdf


$(FILE).pdf: $(FILE).tex content.tex
	#cd figs; make
	pdflatex $(FILE)
	pdflatex $(FILE)

$(FILE)-nup.pdf: $(FILE).pdf
	pdfnup --nup 3x3 $(FILE).pdf

content.tex: content.md
	pandoc -t beamer -Vtheme:Luebeck -Vcolortheme:seahorse content.md -o content.tex

clean:
	rm -f *~ $(FILE).pdf content.tex

edit:
	emacs Makefile $(FILE).tex *.txt *.md &


# eof
