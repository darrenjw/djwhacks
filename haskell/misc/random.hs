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


rollDice :: StdGen -> (Int, StdGen)
--rollDice :: State StdGen Int
rollDice r = randomR (1,6) r :: (Int, StdGen)

-- getStdGen :: IO StdGen -> randomly seed generator at start of a main do block




-- eof

