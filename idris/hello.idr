{-
hello.idr

idris hello.idr -o hello
./hello

-}

module Main


main : IO ()
main = do
  putStrLn "Enter your name"
  name <- getLine
  putStr "Hello, " 
  putStrLn name




-- eof
