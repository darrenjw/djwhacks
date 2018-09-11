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
******.* *.******
*...............*
*.**.*.***.*.**.*
*....*..*..*....*
*.**.*..*..*.**.*
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
  case object GhostBlock extends Block
  case object PacmanBlock extends Block

  type Maze = Vector[Vector[Block]]

  def char2block(x: Char): Block = x match {
    case '*' => Wall
    case '.' => Pill
    case 'p' => PowerPill
    case _ => Empty
  }

  def block2char(b: Block): Char = b match {
    case Wall => '#'
    case Pill => '.'
    case PowerPill => 'o'
    case Empty => ' '
    case GhostBlock => 'M'
    case PacmanBlock => '@'
  }

  val maze0 = mazeLines.toVector.map(_.toVector).map(v => v.map(char2block))

  def pillCount(maze: Maze): Int = maze.map(l => l.map(b => b match {
    case Pill => 1
    case _ => 0
  }).sum).sum

  sealed trait Direction {
    def rand: Direction = {
      val u = math.random
      if (u<0.25) Up
      else if (u<0.5) Down
      else if (u<0.75) Left
      else Right
    }
  }
  case object Up extends Direction
  case object Down extends Direction
  case object Left extends Direction
  case object Right extends Direction

  case class Position(x: Int, y: Int) {
    def move(d: Direction): Position = d match {
      case Up => Position(x,y-1)
      case Down => Position(x,y+1)
      case Left => if (x > 0) Position(x-1,y) else Position(width-1,y)
      case Right => if (x < width-1) Position(x+1,y) else Position(0,y)
    }
  }

  sealed class Sprite(pos: Position, dir: Direction)
  case class Ghost(pos: Position, dir: Direction) extends
      Sprite(pos: Position, dir: Direction)
  case class Pacman(pos: Position, dir: Direction) extends
      Sprite(pos: Position, dir: Direction)

  def updateGhost(m: Maze,g: Ghost): Ghost = {
    val newPos = g.pos.move(g.dir)
    if (m(newPos.y)(newPos.x) == Wall)
      Ghost(g.pos,g.dir.rand)
    else
      Ghost(newPos,g.dir)
  }

  def updatePacman(gs: GameState, key: Int): GameState = {
    val newMaze = gs.m(gs.pm.pos.y)(gs.pm.pos.x) match {
      case Pill => gs.m.updated(gs.pm.pos.y, gs.m(gs.pm.pos.y).updated(gs.pm.pos.x, Empty))
      case _ => gs.m
    }
    val newDir = key match {
      case 97 => Up // A
      case 122 => Down // Z
      case 44 => Left  // ,
      case 46 => Right // .
      case _ => gs.pm.dir
    }
    val newPos = gs.pm.pos.move(newDir)
    val newPacman = if (gs.m(newPos.y)(newPos.x) == Wall)
      Pacman(gs.pm.pos, newDir)
    else
      Pacman(newPos, newDir)
    GameState(newMaze, gs.ghosts, newPacman)
  }

  val ghost0 = Ghost(Position(8,7),Up)
  val ghosts0 = Vector.fill(4)(ghost0)
  val pm0 = Pacman(Position(8,3),Down)

  case class GameState(m: Maze, ghosts: Vector[Ghost], pm: Pacman)

  def updateState(gs: GameState,key: Int): GameState = {
    val newGhosts = gs.ghosts.map(g => updateGhost(gs.m,g))
    updatePacman(GameState(gs.m,newGhosts,gs.pm),key)
  }

  def renderGame(gs: GameState): Unit = {
    val mazeWithGhosts = gs.ghosts.foldLeft(gs.m)((m,g) => m.updated(g.pos.y,m(g.pos.y).updated(g.pos.x,GhostBlock)))
    val completeMaze = mazeWithGhosts.updated(gs.pm.pos.y,mazeWithGhosts(gs.pm.pos.y).updated(gs.pm.pos.x,PacmanBlock))
    val pillsLeft = pillCount(gs.m)
    println("\n\n\n\n\n\nPills left: "+pillsLeft+"\n")
    completeMaze.map(l => l.map(block2char)).map(_.mkString).foreach(println)
    if (pillsLeft == 0) {
      println("\n\n\n *** YOU WIN! ***\n\n\n")
      System.exit(0)
    }
  }


  def main(args: Array[String]): Unit = {
    println(height+" x "+width+" game grid")
    println(pillCount(maze0)+" pills initially")
    val gs0 = GameState(maze0,ghosts0,pm0)

    val con = new jline.console.ConsoleReader
    val is = con.getInput
    val nbis = new jline.internal.NonBlockingInputStream(is,true)
    val charStream = Stream.iterate(0)(x => nbis.read(200))

    charStream.foldLeft(gs0)((gs,key) => {
      val ns = updateState(gs,key)
      renderGame(gs)
      ns
    })

  }

}

// eof
