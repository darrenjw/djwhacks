/*
a-star.scala

A*-search puzzle for FP North East

https://github.com/FP-North-East/atomic-space-helicopters/

https://en.wikipedia.org/wiki/A*_search_algorithm

*/

import scala.collection.immutable.SortedSet

object AStar {

  case class Node(x: Int, y: Int) {
    def allNeighbours: List[Node] = List(Node(x+1,y), Node(x,y+1), Node(x-1,y), Node(x,y-1))
    def neighbours: List[Node] = allNeighbours filter isValid
  }

  val obstacles: Set[Node] = Set(Node(1,0), Node(1,3), Node(1,4), Node(3,2))
  val width: Int = 5
  val height: Int = 4

  def isValid(n: Node): Boolean = n match {
    case Node(x,y) => (x >= 0) & (y >= 0) & (x < width) & (y < height) & !(obstacles.contains(n))
  }





  def main(args: Array[String]): Unit = {
    val start = Node(0,0)
    val target = Node(4,3)
    println(Node(0,0).neighbours)
    println(SortedSet[(Node,Double)]((Node(0,0),0.3),(Node(1,1),0.2),(Node(2,2),0.1))(Ordering.by(_._2)).head)
  }



}

// eof

