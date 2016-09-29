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
	* [ScalaCheck](http://www.scalacheck.org/)
	  * [Intro to property based testing with ScalaCheck](https://blog.codecentric.de/en/2015/11/introduction-to-property-based-testing-using-scalacheck-2)
    * [ScalaIDE](http://scala-ide.org/) for Eclipse
    * [SBT](http://www.scala-sbt.org/) - the scala build tool ([sbt on wikipedia](http://en.wikipedia.org/wiki/SBT_(software)))
	  * [sbt: the missing tutorial](https://github.com/shekhargulati/52-technologies-in-2016/blob/master/02-sbt/README.md)
    * [scalaz](https://github.com/scalaz/scalaz) - functional extensions to core scala (category theory types)
        * [ScalaDoc](http://docs.typelevel.org/api/scalaz/nightly/)
        * [learning scalaz](http://eed3si9n.com/learning-scalaz/)
    * [cats](https://github.com/non/cats) - new alternative to scalaz - will one day replace scalaz, but still a moving target
        * [documentation](http://typelevel.org/cats/)
        * [herding cats](http://eed3si9n.com/herding-cats/) - tutorials
	* [shapeless](https://github.com/milessabin/shapeless) - generic programming extensions to core scala (combinators, hlists, etc.)
	  * [Getting started with shapeless](http://jto.github.io/articles/getting-started-with-shapeless/)
	  * [Shapeless intro and HLists](https://scalerablog.wordpress.com/2015/11/23/shapeless-introduction-and-hlists-part-1/)
	  * [Shapeless HLists](http://enear.github.io/2016/04/05/bits-shapeless-1-hlists/)
	  * [Solving problems generically with shapeless](http://www.cakesolutions.net/teamblogs/solving-problems-in-a-generic-way-using-shapeless)
	  * [Not a tutorial: Part 1](http://kanaka.io/blog/2015/11/09/shapeless-not-a-tutorial-part-1.html)
	  * [Not a tutorial: Part 2](http://kanaka.io/blog/2015/11/10/shapeless-not-a-tutorial-part-2.html)
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
        * [2.0.0 release](http://spark.apache.org/releases/spark-release-2-0-0.html)
        * [Mastering Spark](https://jaceklaskowski.gitbooks.io/mastering-apache-spark/content/)
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
		  * [Diving in to Akka Streams](https://medium.com/@kvnwbbr/diving-into-akka-streams-2770b3aeabb0)
		  * [A first look at Akka Streams](http://rnduja.github.io/2016/03/25/a_first_look_to_akka_stream/)
                  * [Replace actors with Streams](https://softwaremill.com/replacing-akka-actors-with-akka-streams/)
* Web
    * [Lift](http://liftweb.net/) web framework
    * [Play](http://www.playframework.com/) web framework (recommended by [typesafe](http://typesafe.com/))
    * [scala.js](http://www.scala-js.org/) - compile Scala to js for scala on the front-end
	  * [Basic tutorial](http://www.scala-js.org/tutorial/basic/)
	  * [Hands on scala.js](http://www.lihaoyi.com/hands-on-scala-js/#Hands-onScala.js)
	  * [Getting started with scala.js](http://blog.scalac.io/2015/09/24/scala_js.html)
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
		* [A scalable particle filter in Scala](https://darrenjw.wordpress.com/2016/07/22/a-scalable-particle-filter-in-scala/)
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
	* [Pattern match generic types](http://www.cakesolutions.net/teamblogs/ways-to-pattern-match-generic-types-in-scala)
	* [Getting to know CanBuildFrom](http://blog.bruchez.name/2012/08/getting-to-know-canbuildfrom-without-phd.html)
	* [Generalised type constraints](http://blog.bruchez.name/2015/11/generalized-type-constraints-in-scala.html)
	* [Bullet proof data science in Scala](http://www.data-intuitive.com/2016/06/bullet-proof-data-analysis-in-scala/)
    * [Kafka as unix pipes](http://logallthethings.com/2015/09/15/kafka-by-example-kafka-as-unix-pipes/)
    * [Scala one-liners](https://gist.github.com/mkaz/d11f8f08719d6d27bab5)
    * [FP for the average Joe - I - Validation](http://www.47deg.com/blog/fp-for-the-average-joe-part-1-scalaz-validation)
        * [II - Monad transformers](http://www.47deg.com/blog/fp-for-the-average-joe-part-2-scalaz-monad-transformers)
        * [III - Free monads](http://www.47deg.com/blog/fp-for-the-average-joe-part3-free-monads)
* MOOCs
    * [Functional programming](https://www.coursera.org/course/progfun) Scala course on Coursera
        * [Scala cheat sheet](https://github.com/lrytz/progfun-wiki/blob/gh-pages/CheatSheet.md)
    * [Principles of reactive programming](https://www.coursera.org/course/reactive) Follow-up Scala course on Coursera 
* Books
    * [Functional programming in scala](http://www.manning.com/bjarnason/) (non-free p-book, with bundled DRM-free PDF and e-books) - and associated [github repo](https://github.com/fpinscala/fpinscala)


