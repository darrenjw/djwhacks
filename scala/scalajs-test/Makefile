# Makefile

FORCE:
	sbt clean fastOptJS
	make run

run:
	xdg-open ./target/scala-2.13/classes/index.html

edit:
	emacs build.sbt src/main/resources/* src/main/scala/*.scala &




# eof

