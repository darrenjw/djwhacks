# Idris Reminders

```bash
sudo apt-get install haskell-platform
cabal update; cabal install idris


idris
:load hello.idr
:t 4

idris hello.idr
:exec -- run main

idris -o hello hello.idr
./hello


# load the effects package
idris -p effects


```

#### eof

