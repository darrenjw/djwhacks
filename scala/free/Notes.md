# Notes on Free structures

* Church encoding: FP style => OO style
* Reification: OO style => FP style

* OO style is a "shallow embedding"
* FP style is a "deep embedding" - representation as data values on the heap

* Type classes are Church encodings of Free structures
* Free structures are reifications of type classes

* Tagless final is a Church encoding of "data types a la carte" (free monadic coproducts)

* Both Free monads and tagless final encodings solve the "expression problem":

* OO => easy to add "operations" but hard to add "actions" (result types)
* FP => easy to add "actions" but hard to add "operations"

* Coyoneda is just a Free functor for a parameterised type

* Free monads lift any functor into a monad. But needs a functor, because needs to "map". "Coyoneda trick" is to form the free functor of a parameterised type and then lift this into the Free monad. When this gets interpreted into a Monad, there will be an associated Functor with an appropriate map method that can be used.



* Links to various blog posts in source code files...
