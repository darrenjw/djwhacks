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

In state-space modelling, we have a

* forward model: $X_t | x_{t-1} \sim f(x_t|x_{t-1})$ or `f: X => P[X]`
* and observation model: $Y_t|x_t \sim g(y_t|x_t)$ or `g: X => P[Y]`

where `P[_]` is a suitable *probability monad*.

For *filtering* we typically think in terms of predict-update steps:

* $p(x_{t-1}|\mathcal{Y}_{t-1}) \rightarrow p(x_t|\mathcal{Y}_{t-1})$ or `predict: P[X] => P[X]`
* $p(x_t|\mathcal{Y}_{t-1}) \rightarrow p(x_t|\mathcal{Y}_t)$ or `update: (P[X],Y) => P[X]`

`predict` is monadic `flatMap` (or `>>=`) with `f`, and `update` is probabilistic *conditioning* via `g`

Streaming: `advance = update compose predict` where `State = P[X]` - eg. one step of a Kalman or particle filter

# Composable functional models of on-line algorithms and PPLs

* Once we start to think about filtering in terms of operations involving probability monads, it becomes easier to think about how to make models and algorithms composable and scalable, and about the connection to *probabilistic programming* and monadic *probabilistic programming languages* (PPLs)

* Possible to think about all of the standard models and algorithms within this framework: Kalman filters (regular, extended, unscented, ensemble, ...), particle filters (bootstrap, SIR, auxiliary, twisted, ...)

* Simultaneous estimation of (static) parameters and (dynamic) state still problematic: augmented state, Lui & West, Storvik filters, particle learning/practical filters, ... Also on-line (windowed) versions of PMCMC, IBIS, SMC${}^2$, ...

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

Still looking for a really compelling healthcare use-case

