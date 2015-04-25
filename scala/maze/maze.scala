// maze.scala


val width=10
val height=8

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

def extend(tree: Set[Edge]): Set[Edge] = {
 val locs=tree.map{ e => Set(e._1,e._2) }.flatten
 val allEdges=locs.map{neighbours(_)}.flatten
 val newEdges=allEdges.filter { e => {
                 val l=to(e)
                 inrange(l) & !locs.contains(l) 
                 } }
 tree.union(newEdges)
}


// eof


