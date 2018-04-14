{- 
random.idr

idris -p effects
:load random.idr

idris -p effects -o random random.idr && ./random

-}

import Effect.Random

x : Integer
x = 42

-- Need to work through the effects tutorial first...


-- eof

