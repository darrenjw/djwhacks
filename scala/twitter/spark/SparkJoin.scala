/* 
SparkJoin.scala

sbt assembly
/var/tmp/spark-1.4.0-bin-hadoop2.6/bin/spark-submit --master local[4] --driver-memory 2g target/scala-2.10/simple-join-assembly.jar


This works but only actually runs on a single core - need to chunk the tweet file to get it to run across multiple cores...

*/

import java.io.{StringReader,StringWriter}

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import com.github.tototoshi.csv.{CSVParser,CSVReader,CSVWriter}

object SimpleApp {

  def main(args: Array[String]) {

    val mapFile="/home/ndjw1/src/git/djwhacks/scala/twitter/data/mapping.csv"
    //val tweetFile="/home/ndjw1/src/git/djwhacks/scala/twitter/data/first1k.csv"
    val tweetFile="/home/ndjw1/src/git/djwhacks/scala/twitter/data/tweets.csv"
    val outFiles="/home/ndjw1/src/git/djwhacks/scala/twitter/spark/tweets"

    val conf = new SparkConf().setMaster("local").setAppName("Simple Application")
    val sc = new SparkContext(conf)

    // map easy as one line per record - but might be better as broadcast variable...
    //val map=sc.textFile(mapFile).map{CSVParser.parse(_,'\\', ',', '"').get}.map{x=>(x(0),x(1))}
    val map = CSVReader.open(mapFile).allWithHeaders.map{x=>(x("id"),x("username"))}.toMap
    val bMap=sc.broadcast(map)

    // tweets are multiline, so more tricky...
    val tweetBlob=sc.wholeTextFiles(tweetFile)
    val tweets=tweetBlob.flatMap{ case (_,txt) => {
      val reader=CSVReader.open(new StringReader(txt))
      reader.allWithHeaders.map{_.values}
      }
    }

    println("Map count: "+map.size)
    println("Tweet count: "+tweets.count)

    tweets.mapPartitions{ tweets =>
      val stringWriter = new StringWriter()
      val csvWriter = new CSVWriter(stringWriter)
      csvWriter.writeAll((tweets.toList).map{_.toList}.map{x=>bMap.value.getOrElse(x(2),"")::x})
      Iterator(stringWriter.toString)
    }.saveAsTextFile(outFiles)


  }

}



// eof


