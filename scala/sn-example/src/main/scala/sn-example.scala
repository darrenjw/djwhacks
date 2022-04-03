/*
sn-example.scala

Stub for Scala Native with Cats

*/

import cats.*
import cats.implicits.*

object SNCatsApp:

  case class Config(burn: Int = 0, thin: Int = 1, chop: Boolean = false)

  def process(c: Config): Unit =
    var oldLine = "BLANK"
    val iter = scala.io.Source.stdin.getLines
    println(iter.next())
    iter.zipWithIndex.foreach{tup => tup match
      case (line, num) => if ((num >= c.burn)&(num % c.thin == 0))
                            if (c.chop)
                              if (oldLine != "BLANK")
                                println(oldLine)
                              oldLine = line
                            else
                              println(line)
    }

  @main
  def run(args: String*) =
    // Parse command line arguments using "scopt"
    val builder = scopt.OParser.builder[Config]
    val parser =
      import builder._
      scopt.OParser.sequence(
        programName("sn-example"),
        head("sn-example", "0.1"),
        opt[Int]('b', "burn")
          .action((x, c) => c.copy(burn = x))
          .text("lines of burn-in to be removed (default 0)"),
        opt[Int]('t', "thin")
          .action((x, c) => c.copy(thin = x))
          .text("thinning interval (default 1)"),
        opt[Boolean]('c', "chop")
          .action((x, c) => c.copy(chop = x))
          .text("remove final line? (default false)"),
      )
    scopt.OParser.parse(parser, args, Config()) match
      case Some(config) =>
        process(config)
      case _ =>
        println("Parsing of command line arguments failed!")
        System.exit(1)


