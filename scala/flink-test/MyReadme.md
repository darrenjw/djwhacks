# My Readme

In another tab, run
```bash
nc -l 9000
```
and then in _this_ directory run
```bash
sbt "run localhost 9000"
```
Go back to the other tab and enter lines of test text. Check the results back in this tab.

## Run on a flink cluster

Build an assembly jar with:
```bash
sbt assembly
```
Then submit to a flink cluster with something like:
```bash
./bin/flink run ~/....../target/scala-2.12/flink-test-assembly-0.1-SNAPSHOT.jar localhost 9000
```

## sbt template

This template was originally created with:
```bash
sbt new tillrohrmann/flink-project.g8
```
