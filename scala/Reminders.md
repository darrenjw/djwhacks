# Scala reminders

## Versions

* Scala: 2.11.7 (for Java 6/7, or Spark), or 2.12.1 for Java 8
* Sbt: 0.13.8
* Breeze: 0.12 or "latest.integration" for 0.13 snapshot
* Cats: 0.7.0
* Ensime: 1.12.4

## SBT

sbt -h # for help info

* help
* tasks
* settings
* inspect
* reload (reload sbt build definition)

* clean
* compile
* run
* run Arg1 Arg2 ...
* test
* ~test (monitor and re-run when source changes)
* testOnly *blah*
* console
* doc (generate scaladoc in ./target/scala-2.xx/api/)

* projects
* project *blah*
* package (generate jar in ./target/scala-2.xx/)
* publish-local (stuff jar in ivy cache)
* publish (to maven repo)
* assembly (requires assembly plugin)

Supports TAB completion - useful with "test" and "testOnly"

Put "ensime" plugin line in: ~/.sbt/0.13/plugins/plugins.sbt: addSbtPlugin("org.ensime" % "sbt-ensime" % "1.12.4")

## Ensime

From sbt run "ensimeConfig" to create ensime project file.

Then from emacs in a scala buffer, "M-x ensime" to start up ensime

* C-c C-v f - reformat source code
* C-c C-b c - sbt compile
* C-c C-b r - sbt run
* M-x ensime-update - force an update of the ensime server

https://github.com/ensime/ensime-emacs/wiki/Emacs-Command-Reference


## Eclipse (Scala IDE)

From sbt run "eclipse" to create eclipse project files for importing into eclipse

* Shift-Ctrl-F - Reformat file
* Shift-Ctrl-W - Close all windows (from package explorer)
* Shift-Ctrl-P - Go to matching bracket
* Ctrl-Space - Content assist


### Scala worksheet

* Shift-Ctrl-B - Re-run all code


