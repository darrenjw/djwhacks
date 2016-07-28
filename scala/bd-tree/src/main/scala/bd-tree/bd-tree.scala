/*
bd-tree.scala

Simulation of a birth-death tree

 */

object BirthDeath {

  // Some random number support
  import java.util.concurrent.ThreadLocalRandom
  // Use Threadlocalrandom in case of parallelisation at some point
  def runif: Double = ThreadLocalRandom.current().nextDouble()
  def rbern(p: Double): Boolean = (runif < p)
  def rexp(lambda: Double): Double = -math.log(runif) / lambda
  def rdunif(n: Int): Int = ThreadLocalRandom.current().nextInt(n)

  // First simulate the tree using a fast-append data structure
  case class Length(lab: Long, len: Double)
  type LengthTree = Vector[List[Length]]
  val rootSplit = Vector(List(Length(2, 0.0), Length(1, 0.0)), List(Length(3, 0.0), Length(1, 0.0)))

  def extend(lambda: Double, mu: Double)(t: LengthTree): LengthTree = {
    val n = t.length
    val time = rexp(n * (lambda + mu))
    val tt = t map (li => Length(li.head.lab, li.head.len + time) :: li.tail)
    val i = rdunif(n)
    if (rbern(lambda / (lambda + mu))) {
      val lab = tt(i).head.lab
      val ttt = tt.updated(i, Length(2 * lab, 0.0) :: tt(i))
      ttt :+ (Length(2 * lab + 1, 0.0) :: tt(i))
    } else
      tt.updated(i, List(Length(0, -1.0))) filter (_.head.lab > 0) // TODO: Filter is slow!!!
  }

  def treeStream(ext: LengthTree => LengthTree, e: LengthTree): Stream[LengthTree] =
    e #:: treeStream(ext, ext(e))

  def simTreePosEmpty(ext: LengthTree => LengthTree, size: Int): LengthTree =
    treeStream(ext, rootSplit).dropWhile(v => (v.length > 0) & (v.length < size)).head

  @annotation.tailrec
  def simTree(ext: LengthTree => LengthTree, size: Int): LengthTree = {
    val t = simTreePosEmpty(ext, size)
    if (t.length > 0) t else simTree(ext, size)
  }

  // Now we have the tree, map it in to a nicer tree data structure
  trait Tree
  case class Split(len: Double, l: Tree, r: Tree) extends Tree
  case class Leaf(len: Double) extends Tree

  def buildTree(lt: LengthTree): Tree = {
    def formTree(lt: LengthTree, addLength: Double): Tree = {
      if ((lt.length == 1) & (lt(0).length == 1)) Leaf(lt(0).head.len + addLength) else {
        val l1s = lt.map(_.head).toSet
        if (l1s.size > 1) throw new Exception("Not a tree!")
        val l2s = lt.map(_.tail.head).toSet
        val l1sl = l1s.toList
        val l2sl = l2s.toList
        val l2 = lt.map(_.tail)
        if (l2s.size == 1) {
          formTree(l2, l1sl(0).len + addLength)
        } else if (l2s.size == 2) {
          Split(
            l1sl(0).len + addLength,
            formTree(l2 filter (_.head == l2sl(0)), 0.0),
            formTree(l2 filter (_.head == l2sl(1)), 0.0)
          )
        } else {
          throw new Exception("More than 2 splits - not a binary tree!")
        }
      }
    }
    formTree(lt map (_.reverse), 0.0)
  }

  def genTree(ext: LengthTree => LengthTree, size: Int): Tree = buildTree(simTree(ext, size))

  def countLeaves(t: Tree): Int = t match {
    case Leaf(len) => 1
    case Split(len, l, r) => countLeaves(l) + countLeaves(r)
  }

  def treeHeight(t: Tree): Double = t match {
    case Leaf(len) => len
    case Split(len, l, r) => len + math.max(treeHeight(l), treeHeight(r))
  }

  // Now we can generate nice trees, lets think about how to draw them
  import java.awt.image.BufferedImage
  import java.awt.Color
  def drawTree(t: Tree, xsize: Int, ysize: Int): BufferedImage = {
    val bi = new BufferedImage(xsize, ysize, BufferedImage.TYPE_INT_RGB)
    val g = bi.getGraphics
    g.setColor(Color.white)
    g.fillRect(0, 0, xsize, ysize)
    g.setColor(Color.black)
    def putTree(tr: Tree, x: Double, y: Double, width: Double, height: Double): Unit = tr match {
      case Leaf(len) =>
        g.drawLine(x.toInt, y.toInt, (x + width).toInt, y.toInt)
      case Split(len, l, r) => {
        val c = countLeaves(tr)
        val cl = countLeaves(l)
        val cr = countLeaves(r)
        val yl = y + (height / 2) - (height * (1.0 * cl / (cl + cr)) / 2)
        val yr = y - (height / 2) + (height * (1.0 * cr / (cl + cr)) / 2)
        val h = treeHeight(tr)
        val x2 = x + width * (if (h > 0.0) (len / h) else 1.0)
        g.drawLine(x.toInt, y.toInt, x2.toInt, y.toInt)
        g.drawLine(x2.toInt, yl.toInt, x2.toInt, yr.toInt)
        putTree(l, x2, yl, width - width * (if (h > 0.0) (len / h) else 1.0), height * (1.0 * cl / (cl + cr)))
        putTree(r, x2, yr, width - width * (if (h > 0.0) (len / h) else 1.0), height * (1.0 * cr / (cl + cr)))
      }
    }
    putTree(t, 0.05 * xsize, 0.5 * ysize, 0.9 * xsize, 0.9 * ysize)
    bi
  }

  // Main runner function
  def main(args: Array[String]): Unit = {
    println("hi")
    val ext = extend(1.1, 1.0) _
    val tree = genTree(ext, 600)
    println(tree)
    val im = drawTree(tree, 1600, 1200)
    // val im=drawTree(Split(1.0,Leaf(1.0),Split(0.5,Leaf(0.5),Leaf(0.5))),800,600)
    javax.imageio.ImageIO.write(im, "png", new java.io.File("tree.png"))
    println("bye")
  }

}

/* eof */

