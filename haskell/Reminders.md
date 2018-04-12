# Haskell Reminders

## Installation

Debian (including Raspbian) or Ubuntu:
```bash
apt-get install haskell-platform
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


