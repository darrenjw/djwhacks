
# Talk outline

* What is streaming data?
* Computational models for streaming data
* Statistical models for streaming data
* An example: nowcasting urban air pollution
* Streaming Gaussian processes

# Streaming data applications

* Streaming voice and video applications
    * Zoom, Netflix, YouTube, Spotify, etc.
	* Typically streamed directly *over the internet* to your display
	* Even if the video is downloaded to local storage, it is still *streamed* from storage and decompressed on-the-fly for real-time viewing - the entire video is never fully decompressed in RAM - not just about moving data over the internet
* Real-time financial market trading data
    * Automated trading systems
	* Decision support for human traders
* On-line processing (and compression) of scientific experiments
    * Biological sequencing technologies
	* Collider experiments, astronomical surveys
* Real time sensor network data for continuous monitoring
    * Traffic (and pollution) monitoring, weather forecasting, healthcare wearables, ...


# Turing Fellow project

## Streaming data modelling for real-time monitoring and forecasting

* Computational architecture and infrastructure
* Statistical methodology and algorithms

* Case studies: Urban Analytics (The Urban Observatory - eg. air pollution), Healthcare (neuroscience), engineering biology, ...

*Joint work with A Golightly, S Heaps, Y Huang, N Hannaford, A Hardy, ...*

# Flood-PREPARED (NERC-funded)

## Predicting Rainfall Events by Physical Analytics of REaltime Data

* Real-time short-term high-resolution spatio-temporal rainfall modelling, synthesising *areal* weather radar and *point* rain gauge data
* Near real-time emulation of and data assimilation for a hydrodynamic urban flood model
* Hooked up to traffic monitors, CCTV feeds, social media sources, etc., all live streaming (from the Urban Observatory), for the development of an emergency decision support system for Newcastle

**Johnson, Heaps, Wilson, W (2021)** Bayesian spatio-temporal model for high-resolution short-term forecasting of precipitation fields, *arXiv*, 2105.03269

# Streaming data architecture

## Fundamental concepts
* A *stream* is a (possibly infinite) sequence of values of a given (potentially complex) data type, with a definite order
* The stream is accessed one value at a time, and processing is done incrementally, triggered by the arrival of each value
* Typically only *one pass* over the data is possible
* Streams are connected together in a DAG called the *flow graph*

## Software libraries and frameworks
* *Storm*, *Heron*, *Spark streaming*, *Akka streams*, *Flink*, *Kafka streams* are well-known examples of streaming data frameworks
* Frameworks are often run on the *JVM*, built in languages with good support for **functional programming** (FP), such as *Scala*, and rely to a greater-or-lesser extent on FP principles such as *functional reactive programming* (FRP)

# Functional models of streaming data

* A key streaming data processing abstraction is a pure **function**:
$$h: \mathcal{X} \times \mathcal{Y} \longrightarrow \mathcal{X}$$
```
advance: (State, Observation) => State
```
combining current world knowledge (encapsulated in a `State`) together with the latest observation to get an updated world view

* Then given $x_0\in\mathcal{X},\quad \mathbf{y} \in \mathcal{Y}^{\mathbb{N}}$
```
s0: State, sObs: Stream[Observation]
```
we transform the stream of observations $\mathbf{y}$ to states $\mathbf{x}\in\mathcal{X}^{\mathbb{N}}$:
```
sState: Stream[State] = sObs.scan(s0)(advance)
```
via successive application of $h$.

# Stream transformation

Streams are *functors*, since they `map`:
```
  Stream[A].map[B](f: A => B): Stream[B]
```
The `scan` operation, sometimes called `scanLeft`, has signature:
```
  scanLeft[B](init: B)(advance: (B, A) => B): Stream[B]
```
For example:
```
val naturals = Stream.iterate(1)(_ + 1)
// 1, 2, 3, 4, ...
val evens = naturals map (_ * 2)
// 2, 4, 6, 8, ...
val triangular = naturals.scan(0)(_ + _).drop(1)
// 1, 3, 6, 10, ...
```

# State-space modelling

In state-space modelling, we have a

* forward model: $X_t | x_{t-1} \sim f(x_t|x_{t-1})$ or `f: X => P[X]`
* and observation model: $Y_t|x_t \sim g(y_t|x_t)$ or `g: X => P[Y]`

where `P[_]` is a suitable *probability monad*, and $f$, $g$ are *Markov kernels*.

For *filtering* we typically think in terms of predict-update steps:

* $p(x_{t-1}|\mathcal{Y}_{t-1}) \rightarrow p(x_t|\mathcal{Y}_{t-1})$ or `predict: P[X] => P[X]`
* $p(x_t|\mathcal{Y}_{t-1}) \rightarrow p(x_t|\mathcal{Y}_t)$ or `update: (P[X],Y) => P[X]`

`predict` is monadic `flatMap` (or `>>=`) with `f`, and `update` is *probabilistic conditioning* (Bayesian updating) via `g`

Streaming: `advance = update compose predict'` where `State = P[X]` - eg. one step of a Kalman or particle filter

# Composable functional models of on-line algorithms and PPLs

* Once we start to think about filtering in terms of operations involving probability monads and Markov kernels, it becomes easier to think about how to make models and algorithms composable and scalable, and about the connection to *probabilistic programming* and monadic *probabilistic programming languages* (PPLs)

* Possible to think about all of the standard models and algorithms for SSMs within this framework: Kalman filters (regular, extended, unscented, ensemble, ...), particle filters (bootstrap, SIR, auxiliary, ...), etc.

**Law, W (2019)** Functional probabilistic programming for scalable Bayesian modelling, *arXiv*, 1908.02062

# POMP models

* Classical SSMs assume that the data are on a regular equispaced time grid, so that the state evolution model $f(x_t|x_{t-1},\theta)$ represents a single time step of the process
* Many sensors and devices do not generate data on a regular grid, either by design, or due to crashes/reboots creating large gaps of missing values, pushing observations onto a *misaligned grid*, or changes in sampling frequency, etc.
* **Partially observed Markov process** (POMP) models generalise classical SSMs in two important ways:
    * The state evolution model formulated in *continuous time*, and is described by a transition kernel $f(x_{t+t'}|x_t,t',\theta)$
	* It is not (necessarily) required that the transition kernel can be *evaluated* --- only that the state process can by stochastically *simulated* forwards in time

# On-line filtering of POMP models

* The "bootstrap" particle filter is a "likelihood free" algorithm for sequentially computing the filtering distribution of a POMP model (for fixed $\theta$):
$$
\pi(x_t|\mathcal{Y}_t),\ \text{ where } \mathcal{Y}_t \equiv \{y_s|y_s\in\mathcal{Y},s\leq t\}
$$
* Although it is typically presented in discrete time, it works fine for continuous time processes observed discretely at irregular times
* Additionally, composable (and tractable) families of continuous time transition kernels can be built using similar techniques as are sometimes used for discrete time DLMs

**Law \& W (2018)** [Composable models for online Bayesian analysis of streaming data](https://doi.org/10.1007/s11222-017-9783-1), *Statistics and Computing*, **28**:1119-37.

# What makes an algorithm "on-line"?

* Not all streaming data applications are about time series
* Many are just about analysing data based on a single pass
* Almost any statistical algorithm can be expressed in the form of a streaming data algorithm
* All of the data observed so far can be embedded in the *state*, and any analysis whatsoever of the data can be restarted from scratch with the arrival of each new observation!
* We wouldn't consider such an analysis to be *genuinely* on-line
* We typically assume that the "size" of the state is bounded, and that the computational "complexity" of the *advance* step has bounded expectation

# Spatio-temporal modelling

## Spatio-temporal SSMs

* SSMs fit naturally into the streaming data framework
* Can be "on-line", since the **Markov property** for the hidden state process facilitates the bounding of state size and computation associated with updating

## Spatio-temporal GPs

* Relatively straightforward to formulate GPs sequentially and embed in a streaming data framework
* Most commonly used space-time covariance functions don't lead to simple Markov properties, so special techniques for "scalable" and "streaming" GPs must be used to ensure the algorithms are genuinely "on-line"

# Scalable GP modelling

* As the number of observations, $n$, grows, the $n\times n$ covariance (or precision) matrix gradually becomes problematic (whether inversion is explicit or not)
* Can subset or merge design points in more-or-less principled ways, or form some other sparse or low-rank approximation of the covariance (or precision) matrix
* There is also interest in learning GP hyperparameters (such as length scales) in an on-line fashion
* Hybrid approaches using off-line algorithms for learning static (hyper)parameters and on-line algorithms for dynamic state work well in practice
* Very active research area

# Example application: pollution monitoring

## The Urban Observatory: `urbanobservatory.ac.uk`

* The largest set of publicly available real time urban data in the UK --- web API (and also a *websocket* for real time data)
* eg. Temperature, rainfall and air quality sensors around the city
* Rainfall radar data
* *Multivariate*, *spatial*, *temporal*, *irregularly observed*, *mixed modality* (eg. point and areal)

## Pollution mapping in real time

* Pollution monitors at various (fixed) locations around the city
* Measurements every few minutes from every sensor, but not on a fixed grid, and not temporally aligned across sensors
* Would like to "nowcast" a spatially continuous map of pollution levels across the city, updated with each new observation

# Pollution nowcasting

![UO "live" monitoring](uo-gui.png)

# Practical issues

* We have a running testbed system that can visualise pollution levels across the city, using transparent fade-out to represent uncertainty in areas of poor sensor coverage, updating in real-time as each new observation arrives
* Many practical issues requiring further work and collaboration with subject matter experts before public deployment
* Modelling issues
    * Modelling sensor-specific issues and biases --- impact of biases on parameter inference --- especially length scales
	* Incorporation of expert prior information
    * Independent calibration and verification of maps, especially against areal data
* Visualisation issues
    * Colour-scales for inferred pollution levels
	* Appropriate, calibrated visualisation of uncertainty


# Summary

* The analysis and modelling of streaming data is becoming increasingly important
* Typical motivations:
    1. Sequential analysis of "live" data in (near) real time
    2. Analysis of large data-sets based on "one pass" methods
	3. Parallel computation via stream *splitting* and *merging*
* There exist computational models and software libraries for working with streaming data in an efficient and robust way
* Functional (and reactive) programming languages and approaches are well-suited to working with (infinite) data streams
* Time series are a natural fit to streaming data models, but not all streaming data applications have a natural temporal aspect
* Many statistical models and algorithms can be adapted to a sequential context

