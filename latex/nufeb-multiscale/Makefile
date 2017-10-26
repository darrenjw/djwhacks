# Makefile


FORCE:
	make djw-nufeb.pdf

view: djw-nufeb.pdf
	xdg-open djw-nufeb.pdf


djw-nufeb.pdf: djw-nufeb.tex content.tex
	cd figs; make
	pdflatex djw-nufeb
	pdflatex djw-nufeb

content.tex: content.md
	pandoc -t beamer -Vtheme:Luebeck -Vcolortheme:seahorse content.md -o content.tex

clean:
	rm -f *~ djw-nufeb.pdf content.tex

edit:
	emacs Makefile djw-nufeb.tex *.txt *.md &


# eof
