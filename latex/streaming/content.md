# Outline

## Streaming data
* Architecture
* Methodology

## Applications
* Urban Analytics (The Urban Observatory)
* Healthcare (Wearables?)

# Streaming data architecture

* One of the 3 (or 4) "V"s of Big Data is *velocity*
* **Fast data** is all about dealing with newly acquired data in near real-time
* Data engineers are increasingly relying on *streaming data architectures* for managing, routing, analysing and processing of fast data
* *Storm*, *Heron*, *Spark streaming*, *Akka streams*, *Kafka*, *Flink* are well-known examples of streaming data frameworks
* Frameworks are often run on the *JVM*, built in languages with good support for **functional programming** (FP), such as *Scala*, and rely to a greater-or-lesser extent on FP principles such as *functional reactive programming* (FRP)
* Some lesser known libraries take a purer FP approach, such as *Monix* and *FS2*

# Streaming data networks

* There remain interesting CS challenges associated with optimal deployment of computation into heterogeneous streaming data pipelines and networks

## eg. processing smart-watch data
* Activity data measured on smart-watch
* Some summarisation (how much?) before transmitting to smartphone
* Some analysis on smartphone, and some in the Cloud
* How to optimally deploy analysis to optimise for storage, speed and battery life of smart-watch and smartphone (processing and data transmission both consume power)?

# Functional models of streaming data

* The fundamental streaming data abstraction is a **function**:
```
advance: (State, Observation) => State
```
combining current world knowledge (encapsulated in a `State`) together with the latest observation to get an updated world view
* Then given
```
s0: State, sObs: Stream[Observation]
```
we transform the stream of observations to a stream of states:
```
sState: Stream[State] = sObs scanLeft (s0)(advance)
```
* The *definition* of an on-line algorithm is one that can be expressed in terms of a *pure* function, `advance`


# State-space modelling

* In state-space modelling, we have a forward model:

$X_t | x_{t-1} \sim f(x_t|x_{t-1})$ or `f: X => P[X]`

and observation model:

$Y_t|x_t \sim g(y_t|x_t)$ or `g: X => P[Y]`

where `P[_]` is a suitable *probability monad*.

* For *filtering* we typically think in terms of predict-update steps:

$p(x_{t-1}|\mathcal{Y}_{t-1}) \rightarrow p(x_t|\mathcal{Y}_{t-1})$ or `predict: P[X] => P[X]`

$p(x_t|\mathcal{Y}_{t-1}) \rightarrow p(x_t|\mathcal{Y}_t)$ or `update: (P[X],Y) => P[X]`

* `predict` is monadic `flatMap` (or `>>=`) with `f`, and `update` is probabilistic *conditioning* via `g`

* Streaming: `advance = update compose predict` where `State = P[X]` - eg. one step of a Kalman or particle filter

# Composable functional models of on-line algorithms and PPLs

* Once we start to think about filtering in terms of operations involving probability monads, it becomes easier to think about how to make models and algorithms composable and scalable, and about the connection to *probabilistic programming* and monadic *probabilistic programming languages* (PPLs)

* Possible to think about all of the standard models and algorithms within this framework: Kalman filters (regular, extended, unscented, ensemble, ...), particle filters (bootstrap, SIR, auxiliary, twisted, ...)

* Simultaneous estimation of (static) parameters and (dynamic) state still problematic: augmented state, Lui & West, Storvik filters, particle learning/practical filters, ... Also on-line (windowed) versions of PMCMC, IBIS, SMC${}^2$, ...

# Software by CDT students

## Jonny Law

* git.io/statespace - particle filters for composable POMP models
* git.io/dlm - Kalman filters for composable state space models

Scala libraries designed for deployment with Akka Streams

## Tom Cooper

* https://github.com/twitter-archive/caladrius - performance modelling

Python library for Heron

## (Various)

* On-going work developing IoT streaming data solutions using Haskell

# The Urban Observatory

* Newcastle's Urban Observatory project - based in the USB
* www.urbanobservatory.ac.uk
* The largest set of publicly available real time urban data in the UK
* Temperature, rainfall and air quality sensors around the city
* Rainfall radar data
* *Multivariate*, *spatial*, *temporal*, *irregularly observed*, *mixed modality* (eg. point and areal)

## eg. Flood-PREPARED
* NERC Funded project for real-time flood monitoring and prediction for the city based on forecasts and UO data (including rain gauges and rainfall radar)
* Coupling statistical and hydrological models in (near) real-time using data assimiliation

# Healthcare

Big streaming data crops up in numerous problems relating to health

## Wearables
* Currently working on wearables data for Type II diabetes
* Joint modelling of both activity data (from accelerometers) and blood sugar levels (from continuous glucose monitoring devices)
* Developing models for short-term forecasting and alerting
* eg. "Your blood sugar is heading too high - go for a walk around the block"

## Genomics

* eg. analysis of streaming sequencing data in (near) real-time


# Summary

* The on-line analysis of streaming data is becoming increasingly important, especially as the IoT takes off
* Although there are tools for "analytics", so far relatively little effort has gone into the development of robust tools for the *modelling* and *forecasting* of complex quantitative data streams
* State-space modelling techniques from computational statistics and signal processing are well-suited to adoption in streaming data contexts, but some significant challenges remain
* The tooling ecosystem is a bit different to that typically encountered in scientific computing, with more emphasis on the JVM and functional languages, in particular

# Statistics at Newcastle

* **Bayesian methods for big data**: Prof D J Wilkinson, Dr A Golightly, Dr C S Gillespie, Dr J Q Shi, Dr T W M Nye, Dr I J Wilson
* **Time series and nonlinear dynamical systems**: Prof D J Wilkinson, Prof E Porcu, Prof R J Boys, Dr A Golightly, Dr C S Gillespie, Dr S E Heaps, Dr D Prangle, Dr K J Wilson
* **Parallel and distributed statistical computing**: Prof D J Wilkinson, Dr C S Gillespie, Dr D Prangle, Dr I J Wilson
* **Graphical modelling and network inference**: Prof D J Wilkinson, Dr T W M Nye, Dr C J Oates, Dr K J Wilson, Dr S E Heaps
* **Longitudinal and functional data and non-parametric regression**: Prof R Henderson, Dr J Q Shi, Dr W Li
* **Large scale association studies**: Prof H J Cordell, Dr I J Wilson
