{-
vectors.idr

Mess about with Vect

-}

import Data.Vect

x: Vect 4 Int
x = [1, 2, 3, 4]

y: Vect 5 Int
y = 0 :: x

z: Vect 9 Int
z = x ++ y

w: Int
w = index 2 x

xx: Vect 4 (Int, Int)
xx = zip x x

pairSum: (Int,Int) -> Int
pairSum (x,y) = x+y

x2: Vect 4 Int
x2 = map pairSum xx





-- eof

