/*
bd-tree.scala

Simulation of a birth-death tree

 */

object BirthDeath {

  // some random number support
  import java.util.concurrent.ThreadLocalRandom
  def runif: Double = ThreadLocalRandom.current().nextDouble()
  def rbern(p: Double): Boolean = (runif < p)
  def rexp(lambda: Double): Double = -math.log(runif) / lambda
  def rdunif(n: Int): Int = ThreadLocalRandom.current().nextInt(n)

  // First simulate the tree using a fast-append data structure
  case class Length(l: Double)
  type LengthTree = Vector[List[Length]]
  val rootSplit = Vector(List(Length(0.0), Length(0.0)), List(Length(0.0), Length(0.0)))

  def extend(lambda: Double, mu: Double)(t: LengthTree): LengthTree = {
    val n = t.length
    val time = rexp(n * (lambda + mu))
    val tt = t map (li => Length(li.head.l + time) :: li.tail)
    val i = rdunif(n)
    if (rbern(lambda / (lambda + mu))) {
      val ttt = tt.updated(i, Length(0.0) :: tt(i))
      ttt :+ (Length(0.0) :: tt(i))
    } else
      tt.updated(i, List(Length(-1.0))) filter (_.head.l >= 0.0)
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
    def formTree(lt: LengthTree): Tree = {
      if ((lt.length==1)&(lt(0).length==1)) Leaf(lt(0).head.l) else {
        val l1s=lt.map(_.head).toSet
        if (l1s.size>1) println("ERROR: Not a tree!")
        val l2s=lt.map(_.tail.head).toSet
        if (l2s.size==1) {println("ERROR: No split!");println(lt)}
        if (l2s.size>2) println("ERROR: More than 2 splits!")
        val l2sl=l2s.toList
        val l2=lt.map(_.tail)
        Split(l2sl(0).l,formTree(l2 filter (_.head==l2sl(0))),formTree(l2 filter (_.head==l2sl(1))))
        }
      }
    formTree(lt map (_.reverse))
  }

  def main(args: Array[String]): Unit = {
    println("hi")
    val ext = extend(2.0, 0.0) _
    val tree = simTree(ext, 4)
    println(tree.length)
    println(tree)
    val treeTree=buildTree(tree)
    println(treeTree)
    println("bye")
  }

}

/* eof */

