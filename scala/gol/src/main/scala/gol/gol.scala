/*

gol.scala

 */


object Gol {

  import breeze.linalg._
  
  val size=20

  def neighbours(state: DenseMatrix[Int],i: Int,j: Int): Int = {
    ((if (i>0) state(i-1,j) else 0)
    + (if (i<size-1) state(i+1,j) else 0)
    + (if (j>0) state(i,j-1) else 0)
    + (if (j<size-1) state(i,j+1) else 0)
    + (if ((i<size-1)&(j<size-1)) state(i+1,j+1) else 0)
    + (if ((i<size-1)&(j>0)) state(i+1,j-1) else 0)
    + (if ((i>0)&(j<size-1)) state(i-1,j+1) else 0)
    + (if ((i>0)&(j>0)) state(i-1,j-1) else 0))
  }

  def neighbourMatrix(state: DenseMatrix[Int]): DenseMatrix[Int] = 
    DenseMatrix.tabulate(size,size){case (i,j) => neighbours(state,i,j)}

  def nextState(state: DenseMatrix[Int]): DenseMatrix[Int] = {
    val neighMat = neighbourMatrix(state)
    DenseMatrix.tabulate(size,size){case (i,j) => 
      if ((neighMat(i,j)<2)|(neighMat(i,j)>3)) 0
      else if (neighMat(i,j)==3) 1
      else state(i,j)
    }
    }

  @annotation.tailrec
  def genStates(state: DenseMatrix[Int]): Unit = {
    println(state)
    println("")
    Thread.sleep(500)
    genStates(nextState(state))
    }

  def main(args: Array[String]): Unit = {
    println("Hi")
    val initState=DenseMatrix.fill(size,size)(0)
    initState(5,5)=1
    initState(5,6)=1
    initState(5,7)=1
    initState(4,7)=1
    initState(3,6)=1
    genStates(initState)
    }

}


/* eof */


