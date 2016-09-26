

change :: Int -> [Int] -> Int
change = \x c -> if (x==0) then 1 else
  if (length c == 0) then 0 
    else let first = head c
             rest = tail c
         in if (first > x) then change x rest else
      (change (x - first) c) + (change x rest) 

main = print (change (read "12" :: Int) [50,20,10,5,2,1])
