-- This is a single line comment

{- This is a multi-line comment.
   It can span multiple lines.
-}

{- It is possible to {- nest -} multi-line comments -}



import Color exposing (..)
import Graphics.Collage exposing (..)
import Graphics.Element exposing (..)


main : Element
main =
  collage 200 420
    [ move (0,-55) blueSquare
    , move (0, 55) redSquare
    ]


blueSquare : Form
blueSquare =
  traced (dashed blue) square


redSquare : Form
redSquare =
  traced (solid red) square


square : Path
square =
  path [ (50,50), (50,-50), (-50,-50), (-50,50), (50,50) ]



