/* 
puzzle.scala

2, 4, 6, 1, 5 in that order.
Add symbol (from +, -, x, divide) btwn each number to make the end answer 3.
Can use symbol more than once/not at all.

*/

object Puzzle
{

def plus(x: Double, y: Double, v: Boolean) = { if (v==true) println("+") ; x+y}
def minus(x: Double, y: Double, v: Boolean) = { if (v==true) println("-") ; x-y}
def times(x: Double, y: Double, v: Boolean) = { if (v==true) println("*") ; x*y}
def divide(x: Double, y: Double, v: Boolean) = { if (v==true) println("/") ; x/y}

val ops=List(plus _,minus _,times _,divide _)

val poss = for {
  op1 <- ops
  op2 <- ops
  op3 <- ops
  op4 <- ops
  if op4(op3(op2(op1(2,4,false),6,false),1,false),5,false) == 3.0
} yield op4(op3(op2(op1(2,4,true),6,true),1,true),5,true)

// val idx = poss.zipWithIndex.filter(_._1==3.0).head._2


}

// eof


