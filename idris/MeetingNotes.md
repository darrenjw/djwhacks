# Idris


## Edwin Brady - 16/4/18





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

