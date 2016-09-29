# Misc Haskell notes


## Map

```haskell

map:: (alpha -> beta) -> [alpha] -> [beta]
map f [] = []
map f (h:t) = (f h):(map f t)

square:: Int -> Int
square x = x*x

map square [1,2,3] = [1,4,9]

```


