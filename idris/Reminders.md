# Idris Reminders

```bash
# install
sudo apt-get install haskell-platform
cabal update; cabal install idris

# make sure ~/.cabal/bin is in path

idris
:load hello.idr
:t 4

idris hello.idr
:exec -- run main

idris -o hello hello.idr
./hello


# load the effects package
idris -p effects


# generate javascript
idris -o change.js change.idr --codegen javascript


```

#### eof

