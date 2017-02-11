# Makefile

# Use pandoc to convert markdown to other formats


ALLMD=bacillus.md bacilluslabnotes.md darren.md hetero.md motility.md robot-stats.md starterstrains.md yeast.md yeastimaging.md yeastlabnotes.md


FORCE:
	make $(ALLMD)

%.pdf: %.md
	pandoc $< -o $@

%.md: %.docu
	pandoc --from=mediawiki $< -o $@

clean:
	rm -rf *~ $(ALLMD)

# eof





