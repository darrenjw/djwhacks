object Exercise0 {

 import scalaz.Monoid
 import scalaz.syntax.monoid._
 import scalaz.std.string._
 import scalaz.std.anyVal._
 import scalaz.std.option._

 def main(args: Array[String]): Unit = {

  val intResult = 1 |+| 2 |+| mzero[Int]

  def add(items: List[Int]): Int = {
    items.fold(0)(_+_)
  } 

  def addO(items: List[Option[Int]]): Option[Int] = {
    items.fold(Option(0))(_|+|_)
  }
  
  def addItems[A: Monoid](items: List[A]): A =
    items.foldLeft(mzero[A])(_ |+| _)

  val res = addO(List(Some(0),Some(2),Some(3)))
  val res2 = addItems(List(None,Some(2),Some(3)))
  val res3 = addItems(List(some(1),some(2),some(3)))

  println(res)

 }

}
