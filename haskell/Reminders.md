# Haskell Reminders

## Installation

Debian (including Raspbian) or Ubuntu:
```bash
apt-get install haskell-platform
cabal update
```

Get Stack on Ubuntu with (but should now be part of haskell-platform):
```bash
apt-get install haskell-stack
stack update
```

## Using GHC

```bash
ghci
:load myprog.hs
:t myval -- show type of myval

runhaskell myprog.hs

ghc -o myprog myprog.hs
./myprog
```

## Using Stack

```bash
stack new new-project
cd new-project
stack build
stack exec new-project-exe

stack build && stack exec new-project-exe

```


