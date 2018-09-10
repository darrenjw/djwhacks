/*
pacman-console.scala

Simple console version of pacman

*/

object PacmanApp {

  val mazeString = """

*****************
*p......*......p*
*.**.**.*.**.**.*
*...............*
*.**.*.***.*.**.*
*....*..*..*....*
****.*.***.*.****
    ...* *...    
****.*.* *.*.****
*....*.....*....*
*.**.*.***.*.**.*
*...............*
*.**.**.*.**.**.*
*p......*......p*
*****************

"""

  val mazeLines = mazeString.split("\n").filter(_ != "")
  val height = mazeLines.length
  val width = mazeLines.head.length

  sealed trait Block
  case object Wall extends Block
  case object Empty extends Block
  case object Pill extends Block
  case object PowerPill extends Block
  case object Ghost extends Block
  case object Pacman extends Block

  type Maze = Vector[Vector[Block]]

  def char2block(x: Char): Block = x match {
    case '*' => Wall
    case '.' => Pill
    case 'p' => PowerPill
    case _ => Empty
  }

  def block2char(b: Block): Char = b match {
    case Wall => '*'
    case Pill => '.'
    case PowerPill => 'p'
    case Empty => ' '
    case Ghost => 'M'
    case Pacman => '@'
  }

  val maze0 = mazeLines.toVector.map(_.toVector).map(v => v.map(char2block))

  def pillCount(maze: Maze): Int = maze.map(l => l.map(b => b match {
    case Pill => 1
    case _ => 0
  }).sum).sum
  
  def renderMaze(maze: Maze): Unit =
    maze.map(l => l.map(block2char)).map(_.mkString).foreach(println)




  def main(args: Array[String]): Unit = {
    println(height+" x "+width+" game grid")
    println(pillCount(maze0)+" pills initially")
    renderMaze(maze0)
  }

}

// eof
