/* 
TsvFilter.scala

sbt assembly
/var/tmp/spark-1.4.0-bin-hadoop2.6/bin/spark-submit --master local[4] --driver-memory 2g target/scala-2.10/Filter-assembly-0.1.jar


*/

import java.io.{StringReader,StringWriter}

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import com.github.tototoshi.csv.{CSVParser,CSVReader,CSVWriter}

class FQRead(val id: String,val read: String,val qid: String,val qual: String) {

  override def toString: String = s"${id}\n${read}\n${qid}\n${qual}"

  // TODO: this properly would require escaping quotes in the qual field...
  def toJson: String =
    "{\"id\":\"%s\",\"read\":\"%s\",\"qid\":\"%s\",\"qual\":\"%s\"}\n".format(id,read,qid,qual)

  def toTsv: String = s"${id}\t${read}\t${qid}\t${qual}\n"

}

object FQRead {

  def apply(id: String, read: String, qid: String, qual: String): FQRead = new FQRead(id,read,qid,qual)

  def apply(s: String): FQRead = {
    val read=s.split("\t")
    FQRead(read(0),read(1),read(2),read(3))
  }

}



object SimpleApp {

  def main(args: Array[String]) {

    val tname="../genJson/fastq.tsv"
    val outFiles="output"

    val conf = new SparkConf().setMaster("local").setAppName("TSV Filter")
    val sc = new SparkContext(conf)

    val rdd=sc.textFile(tname).map{FQRead(_)}
    println("Unfiltered length: "+rdd.count)

    val frdd=rdd.filter(_.read.take(5)=="TAGCT")
    println("Filtered length: "+frdd.count)
    frdd.saveAsTextFile(outFiles)

  }

}



// eof


