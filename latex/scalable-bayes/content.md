
# Bayesian inference

* Bayesian methods provide a framework for flexible and scalable statistical modelling of data generating processes, and fully probabilistic inference
* Allows separation of the drawing of inferences using data from decision modelling
* For all but the simplest of problems, exact analytical approaches to inference are intractable
* Although optimisation methods can be used for point estimation, Monte Carlo methods have proved to be the most satisfactory approach to fully Bayesian inference
* The Monte Carlo methods that are most effective (eg. MCMC and SMC) are notoriously computationally expensive, and don't always scale or parallelise well

# Scalable Bayesian *modelling*

* Much of the power of the Bayesian approach comes from the fact that most non-trivial Bayesian models are *bespoke*
* Bayesian models are built to carefully capture the underlying data-generating process, and this is especially important in the case of big data, complex data, and/or the synthesis of multiple disparate data sources
* Building large and complex models is challenging, and we need better tools for specifying and exploring such models more conveniently
* *Probabilistic programming languages* (PPLs) allow the separation of model specification from the implementation of an inference algorithm

# Probabilistic programming languages

* Ideally we would specify large models in a PPL, but:
    * Most commonly used PPLs for Bayesian inference are not **compositional** - they don't naturally facilitate the building of a larger model from simpler components
	* The inference algorithms underpinning most PPLs do not scale well to large models or data sets
	* Although compositional PPLs exist, their associated inference algorithms are typically more limited
	* Some exciting recent work on functional PPLs for modular modelling *and* inference, but very early days
* Consequently, most inference algorithms used in practice for problems of non-trivial size and complexity are hand-written and hand-tuned implementations of Monte Carlo algorithms
    * these are difficult to test and debug, and due to lack of separation of concerns, require significant re-writing every time the model changes, greatly limiting model exploration

# Scaling Bayesian inference algorithms

* Challenges in scaling Bayesian inference algorithms in terms of both model complexity ($p$), and number of observations ($n$)
    * Although in theory pure Monte Carlo algorithms can have scaling properties that are independent of $p$, in practice most useful Monte Carlo algorithms don't, and in any case the computational complexity of each iteration will depend on $p$
    * Even in the ideal theoretical case, the per-iteration computational complexity will typically depend linearly on $n$
    * In practice, many commonly used, practically useful algorithms can have $\mathcal{O}(n^2)$ CPU time scaling behaviour (or worse)
* The parallelism story is complex - not easy to get good speed-ups
    * Some MCMC algorithms, such as Gibbs samplers, most naturally partition according to the model structure, while others, such as HMC, partition more naturally over observations

# Big data architectures and functional programming languages

* Functional models of computation are particularly well-suited to parallel and distributed processing of large data sets
* Immutable data structures acted on with pure functions via combinators such as *map* and *reduce* allow automatic parallelisation and distribution of data processing algorithms
    * Many of the difficulties associated with parallel, distributed and concurrent programming involve *shared mutable state* - avoided in functional languages
    * *Hadoop* popularised used the *map-reduce* pattern for batch processing of big data
	* **Apache Spark** uses an in-memory model, vastly improving performance of algorithms requiring multiple passes over the data - *Spark* is written in *Scala*, and exploits lazy evaluation of computational pipelines for optimising data locality, etc.

# Streaming data architecture

* **Fast data** is all about dealing with newly acquired data in near real-time; based on a model of *incremental computation*
* Data engineers are increasingly relying on **streaming data architectures** for managing, routing, analysing and processing of fast data
* *Spark streaming*, *Flink*, *Kafka*, for example, are well-known distributed streaming data frameworks, and the **reactive streams** protocol is supported by several within-node streaming data libraries and frameworks
* Frameworks are often run on the *JVM*, built in languages with good support for **functional programming** (FP), such as *Scala*, and rely to a greater-or-lesser extent on FP principles such as *functional reactive programming* (FRP)
* Very far removed from "traditional HPC"

# Functional models of streaming data

* The fundamental streaming data abstraction is a **function**:
```
advance: (State, Observation) => State
```
combining current world knowledge (encapsulated in a `State`) together with the latest observation to get an updated world view
* Then given `s0: State, sObs: Stream[Observation]` we transform the stream of observations to a stream of states:
```
sState: Stream[State] = sObs scanLeft (s0)(advance)
```
* One possible *definition* of an on-line algorithm is one that can be expressed in terms of a *pure* function, `advance`
* Computation $\mathcal{O}(n)$ in best case, but not if complexity of `advance` depends on $n$ (eg. due to growth of `State`)
* Bayesian inference has a natural sequential formulation

# On-line (Bayesian) filtering of state-space models

In state-space modelling, we have data $\{y_t|t=1,2,\ldots\}$, with a

* forward model: $X_t | x_{t-1} \sim f(x_t|x_{t-1})$ or `f: X => P[X]`
* and observation model: $Y_t|x_t \sim g(y_t|x_t)$ or `g: X => P[Y]`

where `P[_]` is a suitable *probability monad* (perhaps a PPL).

For *filtering* we typically think in terms of predict-update steps:

* $p(x_{t-1}|\mathcal{Y}_{t-1}) \rightarrow p(x_t|\mathcal{Y}_{t-1})$ or `predict: P[X] => P[X]`
* $p(x_t|\mathcal{Y}_{t-1}) \rightarrow p(x_t|\mathcal{Y}_t)$ or `update: (P[X],Y) => P[X]`

`predict` is monadic `flatMap` (or `>>=`) with `f`, and `update` is probabilistic *conditioning* via `g`

Streaming: `advance = update compose predict` where `State = P[X]` - eg. one step of a Kalman or particle filter

# Challenges

* Provision of **hardware infrastructure**: performant, flexible, configurable, elastic (private) cloud computing platforms, at scale and reasonable cost, with good user support
* Lack of sufficient data science **software infrastructure**: algorithms, tools, libraries, and platforms
    * Some reasonable big/fast data platforms and frameworks on which to build, but these aren't perfect, and currently focus mainly on relatively simple analytics and modelling use-cases
* Requirement for highly **multi-disciplinary teams**, due to the broad range of expertise and skills needed
* **Skills and training**
	* Currently most data scientists are completely unaware of the appropriate modern (functional) languages and frameworks needed to scale up, the learning curve is steep, and the academic incentives aren't right
