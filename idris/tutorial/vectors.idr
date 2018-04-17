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

x3: Vect 4 Int
x3 = zipWith (+) x x

m: Vect 3 (Vect 4 Int)
m = [ [1,2,3,4], [5,6,7,8], [9,10,11,12] ]

createEmpties : Vect n (Vect 0 elem)
createEmpties {n = Z} = []
createEmpties {n = (S k)} = [] :: createEmpties

transposem : Vect m (Vect n a) -> Vect n (Vect m a)
transposem Nil = createEmpties
transposem (x :: xs) = zipWith (::) x (transposem xs)



-- eof

