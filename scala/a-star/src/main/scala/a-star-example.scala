/*
a-star-example.scala

A*-search puzzle for FP North East

https://github.com/FP-North-East/atomic-space-helicopters/

https://en.wikipedia.org/wiki/A*_search_algorithm

*/


// The atomic space helicopters example
object AStarExample{

  import AStar._

  // read map from disk
  //val fileName = "field1.json"
  val fileName = "maze2.json"
  val rawJson = scala.io.Source.fromFile(fileName).mkString
  import io.circe._, io.circe.parser._
  val parseResult = parse(rawJson)
  val myMap = parseResult.getOrElse(Json.Null).as[List[List[String]]].
    getOrElse(List(List()))
  val width: Int = myMap.head.length
  val height: Int = myMap.length

  // create a Node type
  case class Node(x: Int, y: Int) {
    def allNeighbours: List[Node] = List(Node(x+1, y),
      Node(x, y+1), Node(x-1, y), Node(x, y-1))
    def neighbours: List[Node] = allNeighbours filter isValid
  }

  def isValid(n: Node): Boolean = n match {
    case Node(x, y) => (x >= 0) && (y >= 0) &&
      (x < width) && (y < height) && (myMap(y)(x) == " ")
  }

  // provide evidence that Node conforms to the GraphNode type class
  implicit val nodeGraphNode = new GraphNode[Node] {
    def neighbours(n: Node): List[Node] = n.neighbours
  }

  // procedure for printing a maze to the console
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

  // main entry point
  def main(args: Array[String]): Unit = {
    printMaze()
    val start = Node(0, 0)
    val target = Node(width - 1, height - 1)
    def h(n: Node): Double = n match {
      case Node(x, y) => (math.abs(x - target.x) +
          math.abs(y - target.y)).toDouble
    }
    val (path, solution) = aStar(start, target, h)
    println(path)
    printMaze(path, solution.closedSet)
  }

}

// eof

