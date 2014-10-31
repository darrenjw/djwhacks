object ScratchTestSheet {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(69); 
  println("Welcome to the Scala worksheet")

  import breeze.io.CSVReader
  import java.io.FileReader
  import breeze.stats.regression._
  import breeze.linalg._;$skip(221); 

  val csv = CSVReader.read(new FileReader("/home/ndjw1/src/git/statslang/scala/data/regression.csv"));System.out.println("""csv  : IndexedSeq[IndexedSeq[String]] = """ + $show(csv ));$skip(24); 
 
  val header = csv(0);System.out.println("""header  : IndexedSeq[String] = """ + $show(header ));$skip(38); 
  val rest = csv.slice(1, csv.length);System.out.println("""rest  : IndexedSeq[IndexedSeq[String]] = """ + $show(rest ));$skip(37); 
  val y = rest map { _(0).toDouble };System.out.println("""y  : IndexedSeq[Double] = """ + $show(y ));$skip(85); 
  val X = rest map { x => (1.0, x(1).toDouble, if (x(2) == "Female") 1.0 else 0.0) };System.out.println("""X  : IndexedSeq[(Double, Double, Double)] = """ + $show(X ));$skip(72); 

  val a = DenseMatrix((1.0, 1.0), (2.0, -2.0), (3.0, 3.0), (4.0, 5.0));System.out.println("""a  : breeze.linalg.DenseMatrix[Double] = """ + $show(a ));$skip(42); 
  val b = DenseVector(2.0, 0.0, 6.0, 9.0);System.out.println("""b  : breeze.linalg.DenseVector[Double] = """ + $show(b ));$skip(34); 
  val result = leastSquares(a, b);System.out.println("""result  : breeze.stats.regression.LeastSquaresRegressionResult = """ + $show(result ));$skip(22); val res$0 = 
  result.coefficients;System.out.println("""res0: breeze.linalg.DenseVector[Double] = """ + $show(res$0));$skip(18); val res$1 = 
  result.rSquared;System.out.println("""res1: Double = """ + $show(res$1));$skip(9); val res$2 = 
  result;System.out.println("""res2: breeze.stats.regression.LeastSquaresRegressionResult = """ + $show(res$2));$skip(42); 
  

  val OI = rest map { _(0).toDouble };System.out.println("""OI  : IndexedSeq[Double] = """ + $show(OI ));$skip(39); 
  val age = rest map { _(1).toDouble };System.out.println("""age  : IndexedSeq[Double] = """ + $show(age ));$skip(65); 
  val sex = rest map { x => if (x(2) == "Female") 1.0 else 0.0 };System.out.println("""sex  : IndexedSeq[Double] = """ + $show(sex ));$skip(183); 
                                                  
                                                  for (i <- 1 to 10) {
                                                  println(i)
                                                  };$skip(164); 
                                                  
val map = scala.collection.mutable.HashMap.empty[Int,String];System.out.println("""map  : scala.collection.mutable.HashMap[Int,String] = """ + $show(map ));$skip(19); val res$3 = 

map+=(1->"hello");System.out.println("""res3: ScratchTestSheet.map.type = """ + $show(res$3));$skip(21); val res$4 = 
map+=(23->"goodbye");System.out.println("""res4: ScratchTestSheet.map.type = """ + $show(res$4));$skip(7); val res$5 = 
map(1);System.out.println("""res5: String = """ + $show(res$5));$skip(8); val res$6 = 
map(23);System.out.println("""res6: String = """ + $show(res$6));$skip(7); val res$7 = 
map(2);System.out.println("""res7: String = """ + $show(res$7));$skip(75); 

val newcsv=CSV("/home/ndjw1/src/git/statslang/scala/data/regression.csv");System.out.println("""newcsv  : <error> = """ + $show(newcsv ))}


}
