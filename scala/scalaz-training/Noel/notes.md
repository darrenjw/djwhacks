#+Title: Today
* Monad transformers
* Case study
* Lunch
* Case study
* Available case studies
** _Pygmy Hadoop_
** Parser combinators
** Validation
** _Event / reactive streams_
** _Free monad interpreters_
** Other random stuff

 Applicative
 join(ES[A])(ES[B])((A, B) => C): ES[C]

 Functor
 map

 Monad
 flatMap(ES[A])(A => ES[B]): ES[B]

 Join
 F[F[A]] => F[A]
