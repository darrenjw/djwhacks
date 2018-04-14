{-
change.idr

Calculate number of ways to make change

-}

module Main

import Data.String

changeCoins : Int -> List Int -> Int
changeCoins 0 c = 1
changeCoins x [] = 0
changeCoins x (f :: r) = if (f > x) then changeCoins x r else
  (changeCoins (x - f) (f :: r)) + (changeCoins x r) 

coins : List Int
coins = [200, 100, 50, 20, 10, 5, 2, 1]

change : Int -> Int
change x = changeCoins x coins

extractInt : Maybe Int -> Int
extractInt Nothing = -1
extractInt (Just x) = x

main : IO ()
main = do
  putStrLn "Enter an amount of money, in pence:"
  amount <- getLine
  let val = parseInteger amount
  let chm = map change val
  let ch = extractInt chm 
  putStr ("Number of ways of making " ++ amount ++ "p is: ")
  printLn ch
  



-- eof
