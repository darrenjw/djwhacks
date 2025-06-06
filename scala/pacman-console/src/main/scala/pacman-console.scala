/*
pacman-console.scala

Simple console version of pacman

Pretty much pure functional. Well, no "vars", anyway... ;-)

*/

object PacmanApp {

  import monocle._
  import monocle.macros._
  import monix.execution.Scheduler.Implicits.global
  import monix.reactive._
  import concurrent.duration._

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
      if (u < 0.25) Up
      else if (u < 0.5) Down
      else if (u < 0.75) Left
      else Right
    }
  }
  case object Up extends Direction
  case object Down extends Direction
  case object Left extends Direction
  case object Right extends Direction

  case class Position(x: Int, y: Int) {
    def move(d: Direction): Position = d match {
      case Up => Position(x, y-1)
      case Down => Position(x, y+1)
      case Left => if (x > 0) Position(x-1, y) else Position(width-1, y)
      case Right => if (x < width-1) Position(x+1, y) else Position(0, y)
    }
  }

  sealed class Sprite(pos: Position, dir: Direction)
  case class Ghost(pos: Position, dir: Direction) extends
      Sprite(pos: Position, dir: Direction)
  case class Pacman(pos: Position, dir: Direction, power: Int, lives: Int) extends
      Sprite(pos: Position, dir: Direction)

  val pacmanLives = GenLens[Pacman](_.lives)

  case class GameState(m: Maze, ghosts: Vector[Ghost], pm: Pacman)

  val gsM = GenLens[GameState](_.m)
  val gsGhosts = GenLens[GameState](_.ghosts)
  val gsPm = GenLens[GameState](_.pm)

  val gsPmLives = gsPm composeLens pacmanLives
  val gsPmLoseLife = gsPmLives.modify(_ - 1)

  def updateGhost(gs: GameState, gi: Int): GameState = {
    val g = gs.ghosts(gi)
    val newPos = g.pos.move(g.dir)
    val newGhost = if (gs.m(newPos.y)(newPos.x) == Wall)
      Ghost(g.pos, g.dir.rand)
    else
      Ghost(newPos, g.dir)
    if ((g.pos == gs.pm.pos)|(newPos == gs.pm.pos)) {
      if (gs.pm.power > 0) {
        gsGhosts.set(gs.ghosts.updated(gi,ghost0))(gs)
      } else {
        gsGhosts.set(gs.ghosts.updated(gi,ghost0))(gsPmLoseLife(gs))
      }
    } else gsGhosts.set(gs.ghosts.updated(gi,newGhost))(gs)
  }

  def updatePacman(gs: GameState, key: Int): GameState = {
    val newPower = if (gs.m(gs.pm.pos.y)(gs.pm.pos.x) == PowerPill) 50 else gs.pm.power
    val newMaze = gs.m(gs.pm.pos.y)(gs.pm.pos.x) match {
      case Pill => gs.m.updated(gs.pm.pos.y, gs.m(gs.pm.pos.y).updated(gs.pm.pos.x, Empty))
      case PowerPill => gs.m.updated(gs.pm.pos.y, gs.m(gs.pm.pos.y).updated(gs.pm.pos.x, Empty))
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
      Pacman(gs.pm.pos, newDir, newPower-1, gs.pm.lives)
    else
      Pacman(newPos, newDir, newPower-1, gs.pm.lives)
    GameState(newMaze, gs.ghosts, newPacman)
  }

  val ghost0 = Ghost(Position(8,7),Up)
  val ghosts0 = Vector.fill(4)(ghost0)
  val pm0 = Pacman(Position(8,3),Down,-1,3)

  def updateState(gs: GameState, key: Int): GameState = {
    val gsu = (0 until 4).foldLeft(gs)((g,i) => updateGhost(g,i))
    updatePacman(gsu,key)
  }

  // TODO: should cancel stream at end of game..
  def renderGame(gs: GameState): Unit = {
    val mazeWithGhosts = gs.ghosts.foldLeft(gs.m)((m,g) => m.updated(g.pos.y,m(g.pos.y).updated(g.pos.x,GhostBlock)))
    val completeMaze = mazeWithGhosts.updated(gs.pm.pos.y,mazeWithGhosts(gs.pm.pos.y).updated(gs.pm.pos.x,PacmanBlock))
    val pillsLeft = pillCount(gs.m)
    println("\n\n\n\nPills left: "+pillsLeft)
    println("Lives: "+gs.pm.lives)
    if (gs.pm.power > 0) println("Power: "+gs.pm.power)
    println("")
    completeMaze.map(l => l.map(block2char)).map(_.mkString).foreach(println)
    if (pillsLeft == 0) {
      println("\n\n\n *** YOU WIN! ***\n\n\n")
      Thread.sleep(5000)	
      System.exit(0)
    }
    if (gs.pm.lives == 0) {
      println("\n\n\n *** YOU LOSE! ***\n\n\n")
      Thread.sleep(3000)
      System.exit(0)
    }
  }


  def main(args: Array[String]): Unit = {
    // initialise game state
    val gs0 = GameState(maze0,ghosts0,pm0)
    // use "jline" to get key presses
    val is = (new jline.console.ConsoleReader).getInput
    val nbis = new jline.internal.NonBlockingInputStream(is,true)
    // use Monix Observable to run the game
    val tickStream = Observable.intervalAtFixedRate(0.2.second)
    val charStream = tickStream.map(i => nbis.read(50))
    val gameStream = charStream.scan(gs0)((gs, key) => updateState(gs,key))
    gameStream.consumeWith(Consumer.foreach(renderGame)).
      runSyncUnsafe(1000.seconds)
  }

}

// eof
