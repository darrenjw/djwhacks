# Makefile

FORCE:
	make omop.pdf

omop.pdf: omop.Rmd
	Rscript -e "rmarkdown::render('omop.Rmd')"

view: omop.pdf
	xdg-open omop.pdf &

edit:
	emacs Makefile abstract.txt links.md *.Rmd &


# eof

