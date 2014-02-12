object bayeskitTest {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(65); 
  println("Welcome to the Scala worksheet")

  import bayeskit.bayeskit._

  import org.apache.commons.math3.distribution._
  
  import breeze.stats.distributions._;$skip(160); val res$0 = 
  // import breeze.linalg._
  
  1 + 2;System.out.println("""res0: Int(3) = """ + $show(res$0));$skip(10); val res$1 = 

  0 to 9;System.out.println("""res1: scala.collection.immutable.Range.Inclusive = """ + $show(res$1));$skip(152); val res$2 = 
  //linspace(1.0,10.0,20)
  
 
  //val cat=new EnumeratedIntegerDistribution((0 to 9).toArray, linspace(0.0,1.0,10).toArray)
  //cat.sample
  

  1 + 1;System.out.println("""res2: Int(2) = """ + $show(res$2));$skip(20); 
  val h = (1, 2, 3);System.out.println("""h  : (Int, Int, Int) = """ + $show(h ));$skip(7); val res$3 = 
  h._1;System.out.println("""res3: Int = """ + $show(res$3));$skip(12); val res$4 = 

  (1, "b");System.out.println("""res4: (Int, String) = """ + $show(res$4));$skip(24); 
  val a = List(1, 2, 3);System.out.println("""a  : List[Int] = """ + $show(a ));$skip(30); 
  val b = List("a", "b", "c");System.out.println("""b  : List[String] = """ + $show(b ));$skip(18); 
  val c = a zip b;System.out.println("""c  : List[(Int, String)] = """ + $show(c ));$skip(74); 

  val state = stepLV(new State(100, 50), 0, 10, Vector(1.0, 0.005, 0.6));System.out.println("""state  : <error> = """ + $show(state ));$skip(105); 

  def fun(a: Int): (Int => Int) = {
    val c = a + 1
    (b: Int) =>
      {
        c + b
      }
  };System.out.println("""fun: (a: Int)Int => Int""");$skip(13); val res$5 = 

  fun(1)(2);System.out.println("""res5: Int = """ + $show(res$5));$skip(32); val res$6 = 

  List(1, 2, 3) zip List(2, 3);System.out.println("""res6: List[(Int, Int)] = """ + $show(res$6));$skip(89); 

  def diff(l: List[Int]): List[Int] = {
    (l.tail zip l) map { x => x._1 - x._2 }
  };System.out.println("""diff: (l: List[Int])List[Int]""");$skip(35); val res$7 = 

  diff(List(1, 2, 3, 4, 6, 7, 9));System.out.println("""res7: List[Int] = """ + $show(res$7));$skip(27); val res$8 = 

  Vector[Double](1.0,2.0);System.out.println("""res8: scala.collection.immutable.Vector[Double] = """ + $show(res$8))}
  




}
