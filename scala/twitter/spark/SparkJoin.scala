/* 
SparkJoin.scala

sbt assembly
/var/tmp/spark-1.3.0-bin-hadoop2.4/bin/spark-submit target/scala-2.10/simple-join-assembly.jar

*/

import java.io.StringReader

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import com.github.tototoshi.csv.{CSVParser,CSVReader}
// CSVParser.parse("a,b,c",'\\', ',', '"').get
// List(a, b, c)


object SimpleApp {

  def main(args: Array[String]) {

    val mapFile="/home/ndjw1/src/git/djwhacks/scala/twitter/data/mapping.csv"
    val tweetFile="/home/ndjw1/src/git/djwhacks/scala/twitter/data/first1k.csv"

    val conf = new SparkConf().setMaster("local").setAppName("Simple Application")
    val sc = new SparkContext(conf)

    // map easy as one line per record
    val map=sc.textFile(mapFile).map{CSVParser.parse(_,'\\', ',', '"').get}.map{x=>(x(0),x(1))}
    
    // tweets are multiline, so more tricky...
    val tweetBlob=sc.wholeTextFiles(tweetFile)
    val tweets=tweetBlob.flatMap{ case (_,txt) => {
      val reader=CSVReader.open(new StringReader(txt))
      reader.allWithHeaders.map{_.values}
      }
    }

    println(tweets.count)

  }

}



// eof


