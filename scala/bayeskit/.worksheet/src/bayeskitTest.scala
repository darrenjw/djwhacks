object bayeskitTest {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(65); 
  println("Welcome to the Scala worksheet")

  import bayeskit.bayeskit._;$skip(39); val res$0 = 

  1 + 1;System.out.println("""res0: Int(2) = """ + $show(res$0));$skip(20); 
  val h = (1, 2, 3);System.out.println("""h  : (Int, Int, Int) = """ + $show(h ));$skip(7); val res$1 = 
  h._1;System.out.println("""res1: Int = """ + $show(res$1));$skip(12); val res$2 = 

  (1, "b");System.out.println("""res2: (Int, String) = """ + $show(res$2));$skip(24); 
  val a = List(1, 2, 3);System.out.println("""a  : List[Int] = """ + $show(a ));$skip(30); 
  val b = List("a", "b", "c");System.out.println("""b  : List[String] = """ + $show(b ));$skip(18); 
  val c = a zip b;System.out.println("""c  : List[(Int, String)] = """ + $show(c ));$skip(81); 

  val state = stepLV(new State(100, 50), 0, 10, new Parameter(1.0, 0.005, 0.6));System.out.println("""state  : bayeskit.bayeskit.State = """ + $show(state ))}

}
