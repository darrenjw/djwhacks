/*
graph test

*/

object GraphTest {

  import scalax.collection.Graph // or scalax.collection.mutable.Graph
  import scalax.collection.GraphPredef._, scalax.collection.GraphEdge._

  def main(args: Array[String]): Unit = {

    println("Hi")
    val g1 = Graph(1, 2, 3, 1 ~ 2, 2 ~ 3)
    println(g1.toString)
    println(g1 get 1)
    println(g1 get 2 neighbors)
  }

}

// eof

