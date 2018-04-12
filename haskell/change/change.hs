{-
change.hs

First stab at a "count change" program in Haskell...
How many different ways can the amount be made using the given coins

ghci
:load change.hs
change 12

runhaskell change.hs

ghc -o change change.hs
./change


-}

changeCoins :: Int -> [Int] -> Int
changeCoins 0 c = 1
changeCoins x [] = 0
changeCoins x (f : r) = if (f > x) then changeCoins x r else
      (changeCoins (x - f) (f : r)) + (changeCoins x r) 

coins = [200,100,50,20,10,5,2,1]

change :: Int -> Int
change x = changeCoins x coins

main = do
  putStrLn "Enter an amount of money (in pence)"
  amountStr <- getLine
  putStrLn ("Number of ways " ++ amountStr ++ "p could be made is:")
  let amount = (read amountStr :: Int)
  --putStrLn (show (change amount))
  print (change amount)

  
  

-- eof



