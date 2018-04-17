# Idris


## Edwin Brady - 16/4/18

* Born in Ashington
* Idris is Pac-man complete (but only have a Space Invaders implementation!)
* *A dependently typed language has first class types*
* https://github.com/edwinb/TypeDD-Samples - samples from the book
* http://www.sicsa.ac.uk/ - scottish informatics and CS alliance
* Type, Define, Refine
* StringOrNat : (isStr : Bool) -> Type
  * A function which takes a boolean as an input and returns a type
  * Nat includes 0
  * StringOrNat False = Nat
  * StringOrNate True = String
* lengthOrDouble : (isStr : Bool) -> StringOrNat isStr -> Nat
* printf in C is sort-of dependently typed - hard-coded in C compiler
* printf "%d %s" 99 "red baloons"
* Vectors!
* dat Vect : Nat -> Type -> Type where
  * Nil : Vect Z a
  * (::): a -> Vect k a -> Vect (S k) a
  * Linked list with a fixed length
* transpose_mat : Vect n (Vect m elem) -> Vect m (Vect n elem)
* --codegen php
* idris -o change.js change.idr --codegen javascript
* Also JavaScript... - distributed as part of the main distribution - well maintained
* State machine with embedded DSL
* Network socket programming example - C API a nightmare!
* Erlang inspired concurrency model - message passing
* JVM backend - Dalvik -> android apps - experiment
* AGDA most similar - but not really general purpose
* 


## Paul Calaghan - 20/6/17

* Dependent type work came out of Durham, late 90s
* Should know that you can't take a head of an empty list
* Vec n A - list of a fixed length n of type A
* (+) : Vec n A -> Vec m A -> Vec (m+n) A
* sort: Vec(n,A) -> Vec(n,A) & sorted(P) [proof that it is sorted]
* A => A - both simple logical statement and function type
* (A=>B) => (B=>C) => (A=C)
* \ f g x -> g (f x)
* Curry-Howard Isomorphism (correspondence)
* v+ : Vec n A -> Vec n A -> Vec n A
* For two vectors of potentially different lengths, can pattern match on lengths and call function from the pattern match on the lengths being the same
* Typeclass DecEq - decidable equality




#### eof

