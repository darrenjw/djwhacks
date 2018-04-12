# Scala reminders

## Versions

* Scala: 2.11.8 (for Java 6/7, or Spark), or 2.12.4 for Java 8
* Sbt: 0.13.16
* Breeze: 0.13
* Cats: 1.0.1
* Ensime: 1.12.4


## SBT

### SBT/Giter8 templates

* `sbt new underscoreio/cats-seed.g8`
* `sbt new darrenjw/breeze.g8`
* `sbt new darrenjw/scala-glm.g8`

### Sbt commands

`sbt -h` # for help info

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

### Misc

* Supports TAB completion - useful with "test" and "testOnly"
* Put "ensime" plugin line in: `~/.sbt/0.13/plugins/plugins.sbt`: `addSbtPlugin("org.ensime" % "sbt-ensime" % "1.12.4")`
* `java -cp $(cat .cp || sbt 'export runtime:fullClasspath' | tail -n1 | tee .cp) <main-class>`


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



#### eof



