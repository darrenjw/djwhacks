# Makefile
# Build slides from Markdown source

MD=slides-test.md
PDF=slides-test.pdf
HTML=slides-test.html

$(HTML): $(MD)
	pandoc -s --mathjax -i -t revealjs -V revealjs-url=http://lab.hakim.se/reveal-js -V theme=sky $(MD) -o $(HTML)

$(PDF): $(MD)
	pandoc -t beamer -Vtheme:Luebeck -Vcolortheme:seahorse $(MD) -o $(PDF)

view: $(PDF)
	xdg-open $(PDF)

view-html: $(HTML)
	xdg-open $(HTML)

clean:
	rm -f $(PDF) $(HTML) *~

# eof

