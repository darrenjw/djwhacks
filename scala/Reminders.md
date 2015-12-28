# Scala reminders

## Versions

* Scala: 2.11.7 (recent version which seems good)
* Breeze: 0.11.2 (contains the thin QR/SVD update)
* Sbt: 0.13.7 (version used in SBT book - blank line requirement dropped)

## SBT

sbt -h # for help info

* help
* tasks
* settings
* inspect
* reload (reload sbt build definition)

* compile
* run
* run Arg1 Arg2 ...
* test
* ~test (monitor and re-run when source changes)
* testOnly
* console
* doc (generate scaladoc in ./target/scala-2.xx/api/)

Supports TAB completion - useful with "test" and "testOnly"

Put "eclipse" plugin in: ~/.sbt/0.13/plugins
Put "gen-ensime" plugin in there, too...

## Ensime

From sbt run "gen-ensime" to create ensime project file.

Then from emacs in a scala buffer, "M-ensime" to start up ensime

* C-c C-v f - reformat source code
* C-c C-b c - sbt compile
* C-c C-b r - sbt run


## Eclipse (Scala IDE)

From sbt run "eclipse" to create eclipse project files for importing into eclipse

* Shift-Ctrl-F - Reformat file
* Shift-Ctrl-W - Close all windows (from package explorer)
* Shift-Ctrl-P - Go to matching bracket
* Ctrl-Space - Content assist


### Scala worksheet

* Shift-Ctrl-B - Re-run all code


