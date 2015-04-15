# Scala reminders

## Versions

* Scala: 2.11.6 (recent version which seems good)
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

## Eclipse (Scala IDE)

* Shift-Ctrl-F - Reformat file
* Shift-Ctrl-W - Close all windows (from package explorer)
* Shift-Ctrl-P - Go to matching bracket
* Ctrl-Space - Content assist


### Scala worksheet

* Shift-Ctrl-B - Re-run all code


