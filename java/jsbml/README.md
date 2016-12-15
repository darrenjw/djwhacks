# jSBML demo, using Maven

For this you need Maven, which on Debian-like systems can be installed with `apt-get install maven`

To build and run the demo, just do

```bash
mvn clean compile exec:java -Dexec.mainClass="DemoApp"
```

Don't need here, but pass in args with: `-Dexec.args="arg0 arg1"`



#### eof


