# Readme.md

## GMRF simulation app

This is actually very simple to build by going right into the source directory containing the `.java` files and then doing:

```bash
javac *.java
java MrfApp
```

However, it's also set up to be build using Maven, from this directory, with something like:

```bash
mvn package
java -jar target/gmrf-1.0-SNAPSHOT.jar
```

#### eof

