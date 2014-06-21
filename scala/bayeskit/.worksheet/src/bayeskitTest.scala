object bayeskitTest {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(65); 
  println("Welcome to the Scala worksheet")

  // shift control b to re evaluate the sheet

  import bayeskit.bayeskit._

  import breeze.stats.distributions._
  import bayeskit.sim._
  import bayeskit.lvsim.stepLV
  import bayeskit.pfilter._
  import bayeskit.pmmh._

  import org.apache.commons.math3.distribution._

  import breeze.stats.distributions._;$skip(350); val res$0 = 
  // import breeze.linalg._

  1 + 2;System.out.println("""res0: Int(3) = """ + $show(res$0));$skip(10); val res$1 = 

  0 to 9;System.out.println("""res1: scala.collection.immutable.Range.Inclusive = """ + $show(res$1));$skip(145); val res$2 = 
  //linspace(1.0,10.0,20)

  //val cat=new EnumeratedIntegerDistribution((0 to 9).toArray, linspace(0.0,1.0,10).toArray)
  //cat.sample

  1 + 1;System.out.println("""res2: Int(2) = """ + $show(res$2));$skip(20); 
  val h = (1, 2, 3);System.out.println("""h  : (Int, Int, Int) = """ + $show(h ));$skip(7); val res$3 = 
  h._1;System.out.println("""res3: Int = """ + $show(res$3));$skip(12); val res$4 = 

  (1, "b");System.out.println("""res4: (Int, String) = """ + $show(res$4));$skip(24); 
  val a = List(1, 2, 3);System.out.println("""a  : List[Int] = """ + $show(a ));$skip(30); 
  val b = List("a", "b", "c");System.out.println("""b  : List[String] = """ + $show(b ));$skip(18); 
  val c = a zip b;System.out.println("""c  : List[(Int, String)] = """ + $show(c ));$skip(33); val res$5 = 

  Vector(1, 2, 3).map { _ * 2 }

  // read a data file
  import scala.io.Source;System.out.println("""res5: scala.collection.immutable.Vector[Int] = """ + $show(res$5));$skip(81); val res$6 = 
  System.getProperty("user.dir");System.out.println("""res6: String = """ + $show(res$6));$skip(49); val res$7 = 

  Source.fromFile("LVpreyNoise10.txt").getLines;System.out.println("""res7: Iterator[String] = """ + $show(res$7));$skip(11); val res$8 = 

  0 to 30;System.out.println("""res8: scala.collection.immutable.Range.Inclusive = """ + $show(res$8))}

}
