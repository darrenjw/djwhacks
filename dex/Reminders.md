# Dex reminders


```bash
dex

dex -h

dex repl

dex script myScript.dx -O

dex --lib-path BUILTIN_LIBRARIES:. script myScript.dx

dex script myScript.dx --outfmt result-only 

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

### Fedora 36

On Fedora there is a slight wrinkle in that clang-12 is no longer part of the standard distribution. Nevertheless, it is relatively easy to download it and build it from source (if somewhat time consuming).

```bash
# Install some packages
sudo dnf install llvm12-devel clang12-libs clang12-devel clang libpng-devel haskell-platform cmake

# Download, build and install clang 12
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-12.0.1/llvm-project-12.0.1.src.tar.xz
tar xvfJ llvm-project-12.0.1.src.tar.xz
cd llvm-project-12.0.1.src
mkdir build
cd build
cmake -G "Unix Makefiles" -DLLVM_ENABLE_PROJECTS="clang" ../llvm
cmake --build .

# Link to clang from somewhere in path
# eg. something like:
ln -s bin/clang-12 ~/.local/bin/clang++-12

# Download, build and install Dex
cd
wget https://github.com/google-research/dex-lang/archive/refs/heads/main.zip
unzip main.zip
cd dex-lang-main
make install
```


