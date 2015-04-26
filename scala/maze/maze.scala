// maze.scala


val width=10
val height=8

val rng = new scala.util.Random

type Loc=(Int,Int)
type Edge=(Loc,Loc)

def neighbours(loc: Loc): Set[Edge] = loc match {
 case (x,y) => Set( ((x,y),(x,y-1)),
                    ((x,y),(x,y+1)),
                    ((x,y),(x-1,y)),
                    ((x,y),(x+1,y)) )
}

def to(edge: Edge): Loc = edge match { case(n1,n2) => n2 }

def inrange(loc: Loc): Boolean = loc match {
 case (x,y) => ( (x>=0) & (y>=0) & (x<width) & (y<height) )
}

def extensions(tree: Set[Edge]): Set[Edge] = {
 val locs=tree.map{ e => Set(e._1,e._2) }.flatten
 val allEdges=locs.map{neighbours(_)}.flatten
 val newEdges=allEdges.filter { e => {
                 val l=to(e)
                 inrange(l) & !locs.contains(l) 
                 } }
 newEdges
}

def pick[T](s: Set[T]):T = s.toList(rng.nextInt(s.size))

@annotation.tailrec def genTree(currTree: Set[Edge]): Set[Edge] = {
 val newEdges=extensions(currTree)
 if (newEdges.size == 0) { currTree } else {
  genTree(currTree + pick(newEdges))
 }
}

// test code

val test=Set( ((0,0),(0,1)) , ((0,1),(1,1)) )
extensions(test)


// eof


