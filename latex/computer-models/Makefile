# Makefile


talk.pdf: talk.tex sim1.pdf lhd.pdf
	pdflatex talk
	pdflatex talk

lhd.pdf: lhd.R
	Rscript lhd.R
	mv Rplots.pdf lhd.pdf

sim1.pdf: gp.R
	Rscript gp.R






view: talk.pdf
	acroread talk.pdf &

edit:
	gedit Makefile talk.tex *.R &


clean:
	rm -f *~ core *.pdf *.ps *.dvi *.log *.aux *.out *.flc

# eof
