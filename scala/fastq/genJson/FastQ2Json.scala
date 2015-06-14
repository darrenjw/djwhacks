/*
FastQ2Json.scala

sbt run

*/

import java.io._
import java.util.zip.GZIPInputStream

class FQRead(id: String, read: String, qid: String, qual: String) {

  override def toString: String = s"${id}\n${read}\n${qid}\n${qual}\n"

  def toJson: String =
    "{\"id\":\"%s\",\"read\":\"%s\",\"qid\":\"%s\",\"qual\":\"%s\"}\n".format(id,read,qid,qual)

}

object FQRead {

  def apply(id: String, read: String, qid: String, qual: String): FQRead = new FQRead(id,read,qid,qual)

  def apply(s: BufferedReader): Option[FQRead] = {
    if (s.ready)
      Some(FQRead(s.readLine,s.readLine,s.readLine,s.readLine))
    else None
  }

}

object FastQ2Json {

  def main(args: Array[String]): Unit = {
    val gzname="Test100k.fastq.gz"
    val in = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(gzname))))
    def fqs: Stream[FQRead] = {
      FQRead(in) match {
        case Some(r) => r #:: fqs
        case None => Stream.empty
      }
    }
    val out = new BufferedWriter(new FileWriter(new File("fastq.json")))
    fqs.foreach{x => out.write(x.toJson)}
    out.close
  }

}

// eof


