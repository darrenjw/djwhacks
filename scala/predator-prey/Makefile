# Makefile

FORCE:
	sbt test package

docs: FORCE
	sbt doc
	cp -r target/scala-2.12/api/* docs/api/
	git add docs/api


Rplots.pdf: analysis.R
	Rscript analysis.R

edit:
	emacs *.R src/main/scala/*.scala src/test/scala/*.scala build.sbt Makefile README.md &



