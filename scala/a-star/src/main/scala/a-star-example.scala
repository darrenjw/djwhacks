/*
a-star-example.scala

A*-search puzzle for FP North East

https://github.com/FP-North-East/atomic-space-helicopters/

https://en.wikipedia.org/wiki/A*_search_algorithm

 */

// The atomic space helicopters example
object AStarExample {

  import AStar._

  case class Maze(width: Int, height: Int, map: List[List[String]])

  def parseMaze(fileName: String): Maze = {
    val rawJson = scala.io.Source.fromFile(fileName).mkString
    import io.circe._, io.circe.parser._
    val parseResult = parse(rawJson)
    val map         = parseResult.getOrElse(Json.Null).as[List[List[String]]].getOrElse(List(List()))
    val width: Int  = map.head.length
    val height: Int = map.length
    Maze(width, height, map)
  }

  // main entry point
  def main(args: Array[String]): Unit = {

    val fileName = if (args.length == 0) "field1.json" else args(0)
    val maze     = parseMaze(fileName)

    // create a Node type
    case class Node(x: Int, y: Int) {
      def allNeighbours: List[Node] =
        List(Node(x + 1, y), Node(x, y + 1), Node(x - 1, y), Node(x, y - 1))
      def neighbours: List[Node] = allNeighbours filter isValid
    }

    def isValid(n: Node): Boolean = n match {
      case Node(x, y) =>
        (x >= 0) && (y >= 0) &&
          (x < maze.width) && (y < maze.height) && (maze.map(y)(x) == " ")
    }

    // provide evidence that Node conforms to the GraphNode type class
    implicit val nodeGraphNode = new GraphNode[Node] {
      def neighbours(n: Node): List[Node] = n.neighbours
    }

    // procedure for printing a maze to the console
    def printMaze(
        path: List[Node] = List(),
        closed: Set[Node] = Set(),
        open: Set[Node] = Set()
    ): Unit = {
      (0 until maze.height).foreach(y => {
        (0 until maze.width).foreach(x => {
          if (maze.map(y)(x) == "#")
            print("#")
          else if (path.contains(Node(x, y)))
            print("X")
          else if (closed.contains(Node(x, y)))
            print("c")
          else if (open.contains(Node(x, y)))
            print("o")
          else
            print(".")
          print(" ")
        })
        println("")
      })
    }

    printMaze()

    val start  = Node(0, 0)
    val target = Node(maze.width - 1, maze.height - 1)

    // Manhattan distance
    def h(n: Node): Double = n match {
      case Node(x, y) =>
        (math.abs(x - target.x) +
          math.abs(y - target.y)).toDouble
    }

    val (path, solution) = aStar(start, target, h)

    println(path)
    printMaze(path, solution.closedSet, solution.openSet.keys.toSet)

  }

}

// eof
