object bayeskitTest {
  println("Welcome to the Scala worksheet")       //> Welcome to the Scala worksheet

  import bayeskit.bayeskit._

  1 + 1                                           //> res0: Int(2) = 2
  val h = (1, 2, 3)                               //> h  : (Int, Int, Int) = (1,2,3)
  h._1                                            //> res1: Int = 1

  (1, "b")                                        //> res2: (Int, String) = (1,b)
  val a = List(1, 2, 3)                           //> a  : List[Int] = List(1, 2, 3)
  val b = List("a", "b", "c")                     //> b  : List[String] = List(a, b, c)
  val c = a zip b                                 //> c  : List[(Int, String)] = List((1,a), (2,b), (3,c))

  val state = stepLV(new State(100, 50), 0, 10, new Parameter(1.0, 0.005, 0.6))
                                                  //> state  : bayeskit.bayeskit.State = bayeskit.bayeskit$State@773cf42a

}