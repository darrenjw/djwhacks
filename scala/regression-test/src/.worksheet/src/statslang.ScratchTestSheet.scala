package statslang

object ScratchTestSheet {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(88); 
  println("Welcome to the Scala worksheet")

  import breeze.io.CSVReader
  import java.io.FileReader
  import breeze.stats.regression._
  import breeze.linalg._;$skip(194); 

  val csv = CSV("/home/ndjw1/src/git/statslang/scala/data/regression.csv");System.out.println("""csv  : statslang.CSV = """ + $show(csv ));$skip(21); 

val head=csv.fields;System.out.println("""head  : List[String] = """ + $show(head ));$skip(24); 

  val age = csv("Age");System.out.println("""age  : breeze.linalg.DenseVector[String] = """ + $show(age ))}

}
