/*
a-star.scala

A*-search puzzle for FP North East

https://github.com/FP-North-East/atomic-space-helicopters/

https://en.wikipedia.org/wiki/A*_search_algorithm

*/


object AStar {

  val rawJson = scala.io.Source.fromFile("field1.json").mkString
  import io.circe._, io.circe.parser._
  val parseResult = parse(rawJson)
  val myMap = parseResult.getOrElse(Json.Null).as[List[List[String]]].getOrElse(List(List()))
  val width: Int = myMap.head.length
  val height: Int = myMap.length

  case class Node(x: Int, y: Int) {
    def allNeighbours: List[Node] = List(Node(x+1,y), Node(x,y+1), Node(x-1,y), Node(x,y-1))
    def neighbours: List[Node] = allNeighbours filter isValid
  }

  def isValid(n: Node): Boolean = n match {
    case Node(x,y) => (x >= 0) && (y >= 0) &&
      (x < width) && (y < height) && (myMap(y)(x) == " ")
  }

  def printMaze(path: List[Node] = List(), closed: Set[Node]=Set()) = {
    (0 until height).foreach(y => {
      (0 until width).foreach(x => {
        if (myMap(y)(x) == "#")
          print("#")
        else if (path.contains(Node(x,y)))
        print("X")
        else if (closed.contains(Node(x,y)))
          print("c")
        else
          print(".")
        print(" ")
        })
        println("")
    })
  }

  case class State(
    cameFrom: Map[Node,Node],
    closedSet: Set[Node],
    openSet: Map[Node, Double],
    gScore: Map[Node,Double]
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
    h: Node => Double,
    target: Node
  ): State = {
    val currentPair = state.openSet.head // TODO: fix to find min
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
          val tentativeGScore = st.gScore(current) + 1.0 // distance between neighbours is 1
          if (tentativeGScore >= st.gScore(ne)) st else {
            State(
              st.cameFrom.updated(ne,current),
              st.closedSet - ne, // correct??
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

    val start = Node(0, 0)
    val target = Node(width-1, height-1)
    //val target = Node(5, 5)
    def h(n: Node): Double = n match {
      case Node(x,y) => (math.abs(x-target.x) + math.abs(y-target.y)).toDouble
    }
    val cameFrom = Map[Node,Node]()
    val closedSet = Set[Node]()
    // val openSet = SortedSet[(Node, Double)]((start, h(start)))(Ordering.by(_._2))
    val openSet = Map[Node, Double](start -> h(start))
    val gScore = Map[Node,Double](start -> 0.0).withDefaultValue(Double.PositiveInfinity)

    val solution = findPath(State(cameFrom, closedSet, openSet, gScore), h, target)
    val path = reconstuctPath(solution.cameFrom, List(target))
    println(path)

    printMaze(path, solution.closedSet)

  }



}

// eof

