/*
maze.scala

 */


object Maze {

//val width=140
//val height=100
val width=40
val height=30

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
 val locs=tree.flatMap{ e => Set(e._1,e._2) }
 val allEdges=locs.flatMap{neighbours(_)}
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

// now create a maze

def main(args: Array[String]): Unit = {  
val init=Set( ((0,0),(0,1)) , ((0,1),(1,1)) )
val tree=genTree(init)

// draw a maze in ascii
def contains(e: Edge): Boolean = e match {
 case (l1,l2) => ((tree.contains(l1,l2))|(tree.contains(l2,l1)))
}

val maze=for {
 y <- 0 until height
 z <- 0 until 2
 x <- 0 until width
 str = if (z==0) {
  if (contains(((x,y),(x,y-1)))) "+ " else "+-"
 } else {
  if (contains(((x,y),(x-1,y)))) "  " else "| "
 }
 termStr = if (x==width-1) {
    if (z==0) str+"+\n" else str+"|\n"
   } 
  else str
} yield termStr

val mazeFinal=maze ++ ( for (x <- 0 until width) yield "+-" )
val mazeStr=mazeFinal.reduce(_+_) + "+\n"
println(mazeStr)

// now draw to an image
import java.awt.image.BufferedImage
import java.awt.{Graphics2D,Color,Font,BasicStroke}
import java.awt.geom._

val wallPx=3
val tunnelPx=10

val widthPx=width*(wallPx+tunnelPx)+wallPx
val heightPx=height*(wallPx+tunnelPx)+wallPx

val canvas = new BufferedImage(widthPx, heightPx, BufferedImage.TYPE_INT_RGB)
val g = canvas.createGraphics()
g.setColor(Color.WHITE)
g.fillRect(0, 0, canvas.getWidth, canvas.getHeight)
g.setColor(Color.BLACK)
// first create all walls
for (x <- 0 to width) {
  g.fillRect(x*(wallPx+tunnelPx),0,wallPx,heightPx)
}
for (y <- 0 to height) {
  g.fillRect(0,y*(wallPx+tunnelPx),widthPx,wallPx)
}
// now remove walls according to the spanning tree
g.setColor(Color.WHITE)
for (e <- tree) {
  val minx=math.min(e._1._1,e._2._1)
  val maxx=math.max(e._1._1,e._2._1)
  val miny=math.min(e._1._2,e._2._2)
  val maxy=math.max(e._1._2,e._2._2)
  g.fillRect(wallPx+minx*(wallPx+tunnelPx),wallPx+miny*(wallPx+tunnelPx),(1+maxx-minx)*tunnelPx,(1+maxy-miny)*tunnelPx)
}
// save image
g.dispose()
javax.imageio.ImageIO.write(canvas, "png", new java.io.File("maze.png"))

}



}


// eof


