/*
ParseTsv.scala

sbt run

*/

import java.io._
//import java.util.zip.GZIPInputStream

class FQRead(id: String, read: String, qid: String, qual: String) {

  override def toString: String = s"${id}\n${read}\n${qid}\n${qual}\n"

  // TODO: this properly would require escaping quotes in the qual field...
  def toJson: String =
    "{\"id\":\"%s\",\"read\":\"%s\",\"qid\":\"%s\",\"qual\":\"%s\"}\n".format(id,read,qid,qual)

  def toTsv: String = s"${id}\t${read}\t${qid}\t${qual}\n"

}

object FQRead {

  def apply(id: String, read: String, qid: String, qual: String): FQRead = new FQRead(id,read,qid,qual)

  def apply(s: BufferedReader): Option[FQRead] = {
    if (s.ready) {
      val read=s.readLine.split("\t")
      Some(FQRead(read(0),read(1),read(2),read(3)))
      }
    else None
  }

}

object ParseTsv {

  def main(args: Array[String]): Unit = {
    val tname="../genJson/fastq.tsv"
    val in = new BufferedReader(new FileReader(tname))
    def fqs: Stream[FQRead] = {
      FQRead(in) match {
        case Some(r) => r #:: fqs
        case None => Stream.empty
      }
    }
    val out = new BufferedWriter(new FileWriter(new File("reads.fastq")))
    fqs.foreach{x => out.write(x.toString)}
    out.close
  }

}

// eof


