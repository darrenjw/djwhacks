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

  def printMaze(path: List[Node] = List()) = {
    (0 until height).foreach(y => {
      (0 until width).foreach(x => {
        if (obstacles.contains(Node(x,y)))
          print("#")
        else if (path.contains(Node(x,y)))
          print("o")
        else
          print(".")
        })
        println("")
    })
  }

  case class State(
    cameFrom: Map[Node,Node],
    closedSet: Set[Node],
    openSet: SortedSet[(Node, Int)],
    gScore: Map[Node,Int]
  )

  @annotation.tailrec
  def reconstuctPath(cameFrom: Map[Node,Node], current: List[Node]): List[Node] = {
    val top = current.head
    if (cameFrom.contains(top))
      reconstuctPath(cameFrom, cameFrom(top) :: current)
    else
      current
  }

  @annotation.tailrec
  def findPath(
    state: State,
    h: Node => Int,
    target: Node
  ): State = {
    val currentPair = state.openSet.head
    val current = currentPair._1
    if (current == target) state else {
      val state1 = State(
        state.cameFrom,
        state.closedSet + current,
        state.openSet.tail,
        state.gScore
      )
      val newState = current.neighbours.foldLeft(state1)((st,ne) => {
        if (st.closedSet.contains(ne)) st else {
          val tentativeGScore = st.gScore(current) + 1 // distance between neighbours is 1
          if (tentativeGScore >= st.gScore(ne)) st else {
            State(
              st.cameFrom.updated(ne,current),
              st.closedSet,
              st.openSet + ((ne, tentativeGScore + h(ne))),
              st.gScore.updated(ne, tentativeGScore)
            )
          }
        }
      })
      findPath(newState,h,target)
    }
  }


  def main(args: Array[String]): Unit = {

    printMaze()

    val start = Node(0,0)
    val target = Node(4,3)
    def h(n: Node): Int = n match {
      case Node(x,y) => math.abs(x-target.x) + math.abs(y-target.y)
    }
    val cameFrom = Map[Node,Node]()
    val closedSet = Set[Node]()
    val openSet = SortedSet[(Node, Int)]((start, h(start)))(Ordering.by(_._2))
    val gScore = Map[Node,Int](start -> 0).withDefaultValue(Int.MaxValue)

    val solution = findPath(State(cameFrom, closedSet, openSet, gScore), h, target)
    val path = reconstuctPath(solution.cameFrom, List(target))
    println(path)

    printMaze(path)

  }



}

// eof

