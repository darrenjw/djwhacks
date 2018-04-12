{-
random.hs

Figure out random stuff in Haskell

Just use pure generators and the State monad for now

Control.Monad.Random is provided by the MonadRandom package, I think

-}

import System.Random
import Control.Monad.State

-- seed a generator
r = mkStdGen 100

-- use generator to construct random values
ir = random r :: (Int, StdGen)

dr = random r :: (Double, StdGen) -- Float is also fine

br = random r :: (Bool, StdGen)

-- dice
dicer = randomR (1,6) r :: (Int, StdGen)

 -- Infinite stream, but don't get the generator back
rdl = take 5 $ randoms r :: [Double]
-- random chars
sr = take 10 $ randomRs ('a','z') r :: [Char]  


-- function to roll a dice
rollDice :: StdGen -> (Int, StdGen)
rollDice r = randomR (1,6) r :: (Int, StdGen)


-- monadic version, using the State monad
rollDiceM :: State StdGen Int
rollDiceM = state (\r -> randomR (1,6) r :: (Int, StdGen)) 

-- function to roll two dice, using a do block
roll2dice :: State StdGen (Int, Int)
roll2dice = do
  roll1 <- rollDiceM
  roll2 <- rollDiceM
  return (roll1, roll2)
  
-- main function to run the dice generator
main :: IO ()
main = do
  rr <- getStdGen -- randomly seed generator at start of a main do block
  print (fst (runState roll2dice rr))
  print (evalState roll2dice rr) -- evalState ditches the state (like runA in Scala/Cats)


-- eof

