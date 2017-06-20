{-
hello.idr

idris hello.idr -o hello
./hello

-}

module Main

main : IO ()
main = putStrLn "Hello world"

