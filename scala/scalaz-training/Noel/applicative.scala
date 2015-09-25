/*
 trait Applicative[F[_]] {
   def zip[A,B,C](fa: F[A])(fb: F[B])(f: (a: A, b: B) => C): F[C]
   def map[A,B](fa: F[A])(f: A => B): F[B]
   def point[A](a: A): F[A]
 }
*/
