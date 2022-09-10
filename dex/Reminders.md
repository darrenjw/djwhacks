# Dex reminders


```bash
dex

dex -h

dex repl

dex script myScript.dx -O

dex --lib-path BUILTIN_LIBRARIES:. script myScript.dx

dex script myScript.dx --outfmt html > myScript.html

dex web myScript.dx

dex --backend llvm script myScript.dx
dex --backend llvm-mc script myScript.dx
dex --backend llvm-cuda script myScript.dx

```

### Working on the compiler

```bash

make install
make tests
make run-tests/linalg-tests
make run-examples/regression


```

## Installation

### Ubuntu

On a clean Ubuntu 22.04 LTS installation, the following sequence of commands should lead to a working Dex installation.

```bash
sudo apt-get update
sudo apt-get install llvm-12-dev clang-12 libpng-dev g++ haskell-platform haskell-stack pkg-config
wget https://github.com/google-research/dex-lang/archive/refs/heads/main.zip
unzip main.zip
cd dex-lang-main
make install
```
Following installation, you may have to log out and back in for the path to get set correctly.

