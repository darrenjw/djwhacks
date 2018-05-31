/*
EvilTraceplots.scala

Some diagnostic plots using EvilPlot

 */


object EvilTraceplots {

  import com.cibo.evilplot.plot._
  import com.cibo.evilplot.colors._
  import com.cibo.evilplot.geometry.Extent
  import com.cibo.evilplot.plot.aesthetics.DefaultTheme._
  import com.cibo.evilplot.numeric.Point


  def traces(out: Seq[Map[String,Double]], truth: Map[String,Double] = Map()): List[List[Plot]] = {
    val keys = out.head.keys.toList.sorted
    keys map (k => {
      val data = out.map(_(k)).zipWithIndex.map(p => Point(p._2,p._1))
      val trace = LinePlot.series(data, "Line graph", HSL(210, 100, 56)).
        xAxis().yAxis().frame().
        xLabel("Iteration").yLabel(k).title("Trace plot")
      val hist = Histogram(out.map(_(k)),30).xAxis().yAxis().frame().
        xLabel(k).yLabel("Frequency")
        truth.get(k) match {
          case None => List(trace, hist)
          case Some(v) => List(trace.hline(v),hist.vline(v))
        }
    })
  }

  def pairs(out: Seq[Map[String,Double]], truth: Map[String,Double] = Map()): List[List[Plot]] = {
    val keys = out.head.keys.toList.sorted
    keys map (k1 => keys map (k2 => {
      if (k1 == k2) {
        val hist = Histogram(out.map(_(k1)),30).xAxis().yAxis().frame().
          xLabel(k1).yLabel("Frequency")
        truth.get(k1) match {
          case None => hist
          case Some(v) => hist.vline(v)
        }
      }
      else if (k1 < k2) {
        val scat = ScatterPlot(out.map(p => Point(p(k1),p(k2)))).
          xAxis().yAxis().frame().
          xLabel(k1).yLabel(k2)
        val scat1 = truth.get(k1) match {
          case None => scat
          case Some(v) => scat.vline(v)
        }
        truth.get(k2) match {
          case None => scat1
          case Some(v) => scat1.hline(v)
        }
      }
      else {
        val cont = ContourPlot(out.map(p => Point(p(k1),p(k2)))).
          xAxis().yAxis().frame().
          xLabel(k1).yLabel(k2)
        val cont1 = truth.get(k1) match {
          case None => cont
          case Some(v) => cont.vline(v)
        }
        truth.get(k2) match {
          case None => cont1
          case Some(v) => cont1.hline(v)
        }
      }
    }))
  }

}

// eof
