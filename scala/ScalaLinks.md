# Scala Programming Links

* General
    * [My Scala tagged bookmarks on delicious](https://delicious.com/darrenjw/scala)
    * [My links on Category Theory and Functional Programming](https://github.com/darrenjw/djwhacks/blob/master/scala/CT-FP.md)
    * [Scala language website](http://www.scala-lang.org)
        * [Documentation](http://www.scala-lang.org/documentation/)
            * [API docs](http://www.scala-lang.org/api/current/) (in !ScalaDoc format)
            * [ScalaDoc](http://docs.scala-lang.org/style/scaladoc.html) style guide
        * [Sourcecode on github](https://github.com/scala/scala)
    * [Scala](http://en.wikipedia.org/wiki/Scala_(programming_language)) on wikipedia
    * [typelevel](http://typelevel.org/) - functional libraries for Scala
    * [ScalaTest](http://www.scalatest.org/user_guide)
    * [ScalaIDE](http://scala-ide.org/) for Eclipse
    * [SBT](http://www.scala-sbt.org/) - the scala build tool ([sbt on wikipedia](http://en.wikipedia.org/wiki/SBT_(software)))
    * [scalaz](https://github.com/scalaz/scalaz) - functional extensions to core scala (category theory types)
        * [ScalaDoc](http://docs.typelevel.org/api/scalaz/nightly/)
        * [learning scalaz](http://eed3si9n.com/learning-scalaz/)
    * [cats](https://github.com/non/cats) - new alternative to scalaz - will one day replace scalaz, but still a moving target
        * [documentation](http://typelevel.org/cats/)
        * [herding cats](http://eed3si9n.com/herding-cats/) - tutorials
    * [shapeless](https://github.com/milessabin/shapeless) - generic programming extensions to core scala (combinators, hlists, etc.)
* Scientific and statistical
    * [Breeze](https://github.com/scalanlp/breeze/) on github (which supercedes [scalala](https://github.com/scalala/Scalala))
    * [Wisp](http://quantifind.com/blog/2015/01/wisp-is-scala-plotting/) - Scala plotting library (web/js)
    * [spire](https://github.com/non/spire) - numeric types (on github)
    * [ScalaR](https://github.com/ScalaR/ScalaR) on github
    * [BIDMat](https://github.com/BIDData/BIDMat) on github (GPU accelerated matrix library for machine learning)
    * [ScalaNLP](http://www.scalanlp.org/)
    * [scalalab](https://code.google.com/p/scalalab/) on google code
    * [jvmr](http://cran.r-project.org/web/packages/jvmr/index.html) - R package for Scala interfacing
    * [rscala](http://dahl.byu.edu/software/rscala/) - replacement for jvmr?
    * Data/data frames/data tables, etc.
        * [saddle](http://saddle.github.io/) - scala data library
        * [scala-csv](https://github.com/tototoshi/scala-csv) - CSV library
        * [scala-datatable](https://github.com/martincooper/scala-datatable) - immutable data table
        * [framian](https://github.com/pellucidanalytics/framian/wiki/Framian-Guide) - another R-like data frame for Scala
    * [Factorie](http://factorie.github.io/factorie/) - probabilistic modelling library
    * [bayes-scala](https://github.com/danielkorzekwa/bayes-scala) - Bayesian networks in scala
    * [List of mathematical tools and libraries](https://wiki.scala-lang.org/display/SW/Tools+and+Libraries#ToolsandLibraries-Mathematics) on the scala wiki
* Big data
    * [Spark](http://spark.apache.org/) - Scalable analytics for scala (from the AMPLab) ([spark on wikipedia](http://en.wikipedia.org/wiki/Spark_(cluster_computing_framework)))
    * [Flink](https://flink.apache.org/) - new Spark alternative (better at streaming? cleaner api?)
    * [Escalante](http://escalante.io/) - Scala for JBoss
    * [Kafka](http://kafka.apache.org/) - stream processing
    * [storm](http://storm-project.net/) - realtime analytics from twitter - now largely obsolete
    * [algebird](https://github.com/twitter/algebird) - abstract algebra (monoid) type from twitter - but spire and cats cover everything?
    * [scalding](https://github.com/twitter/scalding) - cascading for scala from twitter
    * [Akka](http://akka.io/) - distributed computing library based on the [actor model](http://en.wikipedia.org/wiki/Actor_model) ([akka on wikipedia](http://en.wikipedia.org/wiki/Akka_(toolkit)))
        * [Documentation](http://doc.akka.io/docs/akka/2.2.3/scala.html)
        * [Akka Streams](http://doc.akka.io/docs/akka-stream-and-http-experimental/1.0-M4/scala.html)
        * [Scaladoc](http://doc.akka.io/api/akka-stream-and-http-experimental/1.0-M4/) 
* Web
    * [Lift](http://liftweb.net/) web framework
    * [Play](http://www.playframework.com/) web framework (recommended by [typesafe](http://typesafe.com/))
    * [scala.js](http://www.scala-js.org/) - compile Scala to js for scala on the front-end
* Tutorials, blog posts, etc
    * [My blog](http://darrenjw.wordpress.com/)
        * [Gibbs sampler in various languages](http://darrenjw.wordpress.com/2011/07/16/gibbs-sampler-in-various-languages-revisited/)
        * [A functional Gibbs sampler in Scala](http://darrenjw.wordpress.com/2013/10/04/a-functional-gibbs-sampler-in-scala/)
        * [Scala as a platform for statistical computing](http://darrenjw.wordpress.com/2013/12/23/scala-as-a-platform-for-statistical-computing-and-data-science/)
        * [Brief intro to Scala and Breeze for statistical computing](http://darrenjw.wordpress.com/2013/12/30/brief-introduction-to-scala-and-breeze-for-statistical-computing/)
        * [Parallel Monte Carlo using Scala](http://darrenjw.wordpress.com/2014/02/23/parallel-monte-carlo-using-scala/)
        * [Stats languages at the RSS](https://darrenjw.wordpress.com/2014/11/22/statistical-computing-languages-at-the-rss/) - intro to Scala and code examples in a [github repo](https://github.com/darrenjw/statslang-scala)
        * [Calling Scala code from R](https://darrenjw.wordpress.com/2015/01/02/calling-scala-code-from-r-using-jvmr/) (old)
        * [Inlining Scala Breeze code in R](https://darrenjw.wordpress.com/2015/01/03/inlining-scala-breeze-code-in-r-using-jvmr-and-sbt/) (old, but with an update)
        * [Calling R from Scala sbt projects](https://darrenjw.wordpress.com/2015/01/24/calling-r-from-scala-sbt-projects/) (old)
        * [Calling Scala from R](https://darrenjw.wordpress.com/2015/08/15/calling-scala-code-from-r-using-rscala/) (rscala version)
        * [Calling R from Scala](https://darrenjw.wordpress.com/2015/08/15/calling-r-from-scala-sbt-projects-using-rscala/) (rscala version)
        * [Data frames and tables in Scala](https://darrenjw.wordpress.com/2015/08/21/data-frames-and-tables-in-scala/)
        * [HOFs, closures, partial application and currying](https://darrenjw.wordpress.com/2015/11/16/hofs-closures-partial-application-and-currying-to-solve-the-function-environment-problem-in-scala/)
        * [Monads in Scala](https://darrenjw.wordpress.com/2016/04/15/first-steps-with-monads-in-scala/)
    * [Scala school](http://twitter.github.com/scala_school/) (from twitter)
        * [Effective scala](http://twitter.github.io/effectivescala/) (best practices)
    * [A tour of scala](http://docs.scala-lang.org/tutorials/tour/tour-of-scala.html)
    * [The Neophyte's Guide to Scala ](http://danielwestheide.com/scala/neophytes.html)
    * [Learning scalaz](http://eed3si9n.com/learning-scalaz/), including a [scalaz cheatsheet](http://eed3si9n.com/learning-scalaz/scalaz-cheatsheet.html)
    * [Scala overview on stack overflow](http://stackoverflow.com/tags/scala/info)
    * [Programming in Scala](http://www.artima.com/pins1ed/) (first edition, on-line)
    * [Scala glossary](http://docs.scala-lang.org/glossary/)
    * [Scala by example](http://www.scala-lang.org/docu/files/ScalaByExample.pdf) (PDF)
    * [Design patterns in scala](http://pavelfatin.com/design-patterns-in-scala/)
    * [Kafka as unix pipes](http://logallthethings.com/2015/09/15/kafka-by-example-kafka-as-unix-pipes/)
    * [Functional programming](https://www.coursera.org/course/progfun) Scala course on Coursera
        * [Scala cheat sheet](https://github.com/lrytz/progfun-wiki/blob/gh-pages/CheatSheet.md)
    * [Principles of reactive programming](https://www.coursera.org/course/reactive) Follow-up Scala course on Coursera 
    * [Functional programming in scala](http://www.manning.com/bjarnason/) (non-free p-book, with bundled DRM-free PDF and e-books) - and associated [github repo](https://github.com/fpinscala/fpinscala)


