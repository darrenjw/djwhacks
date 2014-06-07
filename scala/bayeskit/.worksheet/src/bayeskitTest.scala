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
  val c = a zip b;System.out.println("""c  : List[(Int, String)] = """ + $show(c ));$skip(71); 

  val state = stepLV(Vector(100, 50), 0, 10, Vector(1.0, 0.005, 0.6));System.out.println("""state  : bayeskit.sim.State = """ + $show(state ));$skip(105); 

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

  diff(List(1, 2, 3, 4, 6, 7, 9));System.out.println("""res7: List[Int] = """ + $show(res$7));$skip(28); val res$8 = 

  Vector[Double](1.0, 2.0);System.out.println("""res8: scala.collection.immutable.Vector[Double] = """ + $show(res$8));$skip(235); 

  def simPrior(n: Int, t: Time, th: Parameter): Vector[State] = {
    val prey = new Poisson(100.0).sample(n).toVector
    val predator = new Poisson(50.0).sample(n).toVector
    prey.zip(predator) map { x => Vector(x._1, x._2) }
  };System.out.println("""simPrior: (n: Int, t: bayeskit.sim.Time, th: bayeskit.sim.Parameter)Vector[bayeskit.sim.State]""");$skip(109); 
  def obsLik(s: State, o: Observation, th: Parameter): Double = {
    new Gaussian(s(0), 10.0).pdf(o(0))
  };System.out.println("""obsLik: (s: bayeskit.sim.State, o: bayeskit.sim.Observation, th: bayeskit.sim.Parameter)Double""");$skip(82); 
  val truth = simTs(Vector(100, 50), 0, 30, 2.0, stepLV, Vector(1.0, 0.005, 0.6));System.out.println("""truth  : bayeskit.sim.StateTS = """ + $show(truth ));$skip(65); 
  val data = truth map { x => (x._1, Vector(x._2(0).toDouble)) };System.out.println("""data  : List[(bayeskit.sim.Time, scala.collection.immutable.Vector[Double])] = """ + $show(data ));$skip(62); 
  val mll = pfMLLik(100, simPrior, 0.0, stepLV, obsLik, data);System.out.println("""mll  : bayeskit.sim.Parameter => Option[Double] = """ + $show(mll ));$skip(47); 
  val mllSample = mll(Vector(1.0, 0.005, 0.6));System.out.println("""mllSample  : Option[Double] = """ + $show(mllSample ));$skip(67); 

  val pmll = pfMLLikPar(100, simPrior, 0.0, stepLV, obsLik, data);System.out.println("""pmll  : bayeskit.sim.Parameter => Option[Double] = """ + $show(pmll ));$skip(49); 
  val pmllSample = pmll(Vector(1.0, 0.005, 0.6));System.out.println("""pmllSample  : Option[Double] = """ + $show(pmllSample ));$skip(32); val res$9 = 

  mll(Vector(1.0, 0.005, 0.6));System.out.println("""res9: Option[Double] = """ + $show(res$9));$skip(31); val res$10 = 
  mll(Vector(1.0, 0.005, 0.6));System.out.println("""res10: Option[Double] = """ + $show(res$10));$skip(31); val res$11 = 
  mll(Vector(1.0, 0.005, 0.6));System.out.println("""res11: Option[Double] = """ + $show(res$11));$skip(33); val res$12 = 

  pmll(Vector(1.0, 0.005, 0.6));System.out.println("""res12: Option[Double] = """ + $show(res$12));$skip(32); val res$13 = 
  pmll(Vector(1.0, 0.005, 0.6));System.out.println("""res13: Option[Double] = """ + $show(res$13));$skip(32); val res$14 = 
  pmll(Vector(1.0, 0.005, 0.6));System.out.println("""res14: Option[Double] = """ + $show(res$14));$skip(26); val res$15 = 

  Vector(1,2,3).map{_*2}

 // read a data file
 import scala.io.Source;System.out.println("""res15: scala.collection.immutable.Vector[Int] = """ + $show(res$15));$skip(78); val res$16 = 
 System.getProperty("user.dir");System.out.println("""res16: String = """ + $show(res$16));$skip(48); val res$17 = 

 Source.fromFile("LVpreyNoise10.txt").getLines;System.out.println("""res17: Iterator[String] = """ + $show(res$17));$skip(14); val res$18 = 
 
 
  0 to 30;System.out.println("""res18: scala.collection.immutable.Range.Inclusive = """ + $show(res$18))}
  
  
}
