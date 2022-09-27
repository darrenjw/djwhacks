# Makefile

#FILE=djw-streaming
#FILE=djw-turing
FILE=djw-liverpool


CONTENT=$(FILE)-content

FORCE:
	make $(FILE).pdf

view: $(FILE).pdf
	xdg-open $(FILE).pdf &

present: $(FILE).pdf
	okular $(FILE).pdf &

$(FILE).pdf: $(FILE).tex $(CONTENT).tex
	#cd figs; make
	pdflatex $(FILE)
	pdflatex $(FILE)

$(FILE)-nup.pdf: $(FILE).pdf
	pdfnup --nup 3x3 $(FILE).pdf

%.tex: %.md
	pandoc -t beamer -Vtheme:metropolis $< -o $@

clean:
	rm -f *~ $(FILE).pdf *.nav *.aux

edit:
	emacs Makefile $(FILE).tex *.txt *.md &


# eof
