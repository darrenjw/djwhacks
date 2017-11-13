
# Outline

* The NUFEB project
* NUFEB exemplar: Wastewater treatment plants
* Modelling across scales
* Microscale modelling of complex bacterial community growth
* Statistical emulation of the microscale model
* Mesoscale modelling and emulation as an upscaling strategy

# The NUFEB project

* NUFEB: *Newcastle University Frontiers in Engineering Biology*
* `research.ncl.ac.uk/nufeb/`
* £5.5M EPSRC Frontiers in Engineering award
* Roughly 12 investigators at Newcastle, across Engineering, Maths & Stats, Computing Science, similar number of RAs
* Collaboration with *Northumbrian Water* --- `www.nwl.co.uk`
* Project title:

**A new frontier in design:**

**the simulation of open engineered biological systems**

* Concerned with the multi-scale stochastic simulation of complex biological communities in an open environmental system, with a view to using simulation for designing engineering interventions to improve performance
* Exemplar application: Wastewater treatment systems

# NUFEB exemplar: Wastewater treatment plants

* Wastewater treatment plants (WTPs) are an ideal example of a multi-scale open engineered biological system
* There are roughly $10^{18}$ individual bacteria in a typical plant
* Macro-scale characteristics of WTPs are the consequence of micro-scale features of a vast number of individual bacteria acting together in complex ecological communities
* The properties of cells at the micro-scale level dictate the behaviour of a WTP at the macro scale
* Need to model and understand the interaction of microbes at a fine scale to accurately capture micro-scale responses
* Need to transfer this fine-scale information to the engineered macroscale process in a computationally efficient way

# Two different water treatment technologies

Micro-scale modelling similar, but different up-scaling strategies

*From Wikipedia:*

## Activated sludge systems

Treating wastewater in large tanks using aeration and a biological floc composed of bacteria and protozoa

## Trickling filter

Consists of a fixed bed of rocks, lava, coke, gravel, or plastic, or other media over which sewage or other wastewater flows downward and causes a layer of microbial slime (biofilm) to grow, covering the bed of media

Aerobic conditions are maintained by splashing, diffusion, and usually by natural convection of air if the filter medium is porous

# Data and measurements for model validation and calibration

## Micro-scale

We have laboratory facilities for growing and measuring biological, chemical and mechanical properties of flocs and biofilms

## Bench-scale

We can run bench-scale experiments in chemostats and similar devices for testing our meso-scale upscaling

## Pilot plant

We also have a pilot plant facility for experiments at the macro-scale

# BE:WISE

**Biological Engineering: Wastewater Innovation at Scale**

* Pilot plant facility for model validation and testing
* The largest wastewater treatment research facility in Europe, developing innovations for sustainable wastewater treatment
* Funded by a £1.2 million grant from the EPSRC, and with funding from Newcastle University and Northumbrian Water, BE:WISE is an international centre of research excellence and demonstration for wastewater management technology
* The BE:WISE facility is made available to academic researchers worldwide.
* `research.ncl.ac.uk/bewise`

# Modelling across scales

![Modelling across scales](figs/olu-003.png){height=60%}


# Microscale modelling of bacterial community growth

![Overview of microscale IB model](figs/jaya-004.png){height=60%}

# Biological/mechanical modelling

* Each bacterial cell is an agent within an individual based model of cell growth, division, death and mechanical interactions
* First order growth kinetics, solved numerically using an explicit Euler scheme
* Growth rates depend on the nutrient environment of the agent, and nutrients are taken up from the fluid during growth
* Cell division occurs when cells reach a critical size --- stochastic elements in the division process
* EPS grows initially as a shell around appropriate cells, then excretes as a discrete particle/agent (placed randomly)
* Mechanical relaxation is required due to cell growth and division --- a discrete element method (DEM) is used, solving for each particle in the Lagrangian framework --- springs and dashpot model --- using LAMMPS
* The net force on each agent is the sum of contact, adhesion, and fluid forces

# Mechanics

![Mechanical interactions](figs/jaya-008.png){height=60%}

# Physical/chemical modelling

* Nutrient mass-balance --- advection--diffusion--reaction equation, for each nutrient, $S$:
$$\frac{\partial S}{\partial t} + \vec{U}\cdot\nabla S = \nabla\cdot(D_e\nabla S) + R$$
* $\vec{U}$ is fluid velocity, $D_e$ effective diffusion coefficient, and $R$ nutrient uptake rate
* Rectangular computational domain and discretisation
* Temporal derivatives forward Euler
* Spatial derivatives central finite differences
* Due to time scale separation between physical and biological processes, we typically solve to steady state within each biological time step

# Computational domain

![Computational domain](figs/olu-005.png){height=60%}


# Interaction of multiple bacterial species

![HET, AOB and NOB interaction](figs/jaya-011.png){height=60%}

# Nutrient abundance and diffusion

![Biofilm growth under differing nutrient conditions](figs/jaya-015.png){height=60%}

# Flocs and biofilms

![Floc and biofilm growth under differing nutrient conditions](figs/olu-004.png){height=60%}

# Varying boundary conditions

![Smooth biofilm growth when nutrients are supplied from below](figs/jaya-021.png){height=50%}

# Biofilm growth in a shear fluid flow

![Shear flow smooths a rough biofilm](figs/jaya-022.png){height=60%}

# Deformation and detachment

![Shear flow deformation, streamers, detachment](figs/jaya-026.png){height=60%}

# NUFEB IB Models

## NUFEB 1

* The NUFEB 1 model is free open source software on GitHub, built on top of LAMMPS: https://github.com/nufeb/NUFEB

## NUFEB 2

* The NUFEB 2 model is in use internally, and will be released on publication
* Microbes and nutrients: many microbial species, gas and liquid nutrients
* Biological features: Monod-based growth, energy-based growth, cell division, EPS production, cell death
* Chemical features: pH, gas-liquid transfer, thermodynamics
* Physical features: nutrient mass balance (diffusion), EPS adhesion, shear force, contact force, viscous force

# Scaling up

* The model can run up to order $10^6$ bacteria
* Typically run for cubic volumes with side 200um (though particle numbers are the main limiting factor)
* Current collaboration with Oak Ridge National Laboratory (ORNL, www.ornl.gov) on parallelisation of the code to run effectively on the ORNL supercomputer (and also some wet lab work for model validation)
* Regardless, unlikely to scale to more than $10^8$ or $10^9$ bacteria in practice
* Still a long way off the $10^{18}$ bacteria in a typical full-scale WTP...

# Statistical emulation of complex computer models

* The micro-scale model is very computationally intensive --- approx 1 CPU day to simulate 5 real days
* There now exists a large literature on the use of statistical model *emulators* in the design and analysis of computer experiments
* Emulators essentially provide a fast approximate surrogate for the true model, with properly quantified uncertainty, but have a number of different possible uses, including:
1. History matching (finding inputs which give output consistent with data)
2. Calibration (finding parameter settings most consistent with data)
3. Model validation (deciding whether the model is a good fit to reality)
4. Design (finding inputs leading to a desirable output)

# Gaussian process emulation

* A GP is a probability distribution on functions defined so that the marginal distribution of any finite number of points always has a multivariate normal (MVN) distribution
* Points close together in input space are typically more highly correlated than points far away
* Stationary GPs are defined by a *covariance function* --- many different possible choices --- a Gaussian kernel is popular
$$
\operatorname{Cov}[f(\mathbf{x}),f(\mathbf{x}')] = K(\mathbf{x},\mathbf{x}') = \sigma^2\exp\left\{-\left(\frac{\Vert \mathbf{x}-\mathbf{x}'\Vert_2}{r}\right)^2\right\},
$$
containing two parameters: an asymptotic variance, $\sigma^2$, and a correlation length scale, $r$.
* A GP conditioned on observations is also a GP (*Kriging*)

# Sequential updating of a GP emulator

![GP conditioning](figs/emu1.pdf){height=60%}

# Sequential updating of a GP emulator

![GP conditioning](figs/emu2){height=60%}

# Sequential updating of a GP emulator

![GP conditioning](figs/emu3){height=60%}

# Sequential updating of a GP emulator

![GP conditioning](figs/emu4){height=60%}

# Sequential updating of a GP emulator

![GP conditioning](figs/emu5){height=60%}

# Sequential updating of a GP emulator

![GP conditioning](figs/emu6){height=60%}


# Samples from a GP posterior distribution

![GP samples](figs/sample1){height=60%}

# Samples from a GP posterior distribution

![GP samples](figs/sample10){height=60%}

# Samples from a GP posterior distribution

![GP samples](figs/sample100){height=60%}

# Design of computer experiments

* To build a "good" emulator, we want residual uncertainty to be small. In 1d this is easy, but in higher dimensions we need to choose *design points* to *fill space* efficiently so that there aren't big gaps in parameter space for which we don't have a simulator run
* The naive approach to this problem is to construct a *Cartesian product design* where many levels of each variable are considered in turn, but this approach becomes unworkable in more than 2 or 3 dimensions
* *Latin hypercube designs* (LHDs) are a good way to choose design points to fill space in an efficient way
* In more than 2 dimensions, Cartesian product designs are a very inefficient way of covering the design space, and LHDs are much better. In other words, naive *parameter scans are bad --- don't do them!*

# A 2D Latin hypercube design

![2D LHD](figs/lhd){height=60%}

# Statistical emulation of the microscale model

* We typically emulate important univariate summary statistics arising from the analysis of the raw computer model output (eg. Floc diameter, floc mass, biofilm height, surface roughness, community diversity index, ...) based on numerous inputs (covariates)
* Consider output $y$ for input $\mathbf{x}$ as a deterministic function
$$y = f(\mathbf{x})$$
* Outputs $y_i$ at $n$ design points $\mathbf{x}_i$ statistically modelled as
$$ \mathbf{y} \sim \mathcal{N}(\mathsf{H}\boldsymbol{\beta},\mathsf{A}) $$
where $\mathsf{H}$ is a design matrix where the $i$th row of $\mathsf{H}$ is a (deterministic) function of $\mathbf{x}_i$, and $\mathsf{A}$ is the matrix of covariances determined by the Gaussian kernel described earlier

# Multivariate emulation

* Univariate emulation ignores correlation between multiple outputs --- often better to jointly model
* Covariance matrix for $p$ outputs a $p\times p$ matrix $\Sigma$
* Assuming separability of the two covariance matrices, we get a matrix normal distribution for $n\times p$ output matrix $\mathsf{Y}$
$$ \mathsf{Y} \sim \mathcal{MN}(\mathsf{H}\mathsf{B},\mathsf{A},\Sigma) $$
* ie. $\operatorname{Var}(\boldsymbol{\operatorname{vec}} Y) = \Sigma \otimes \mathsf{A}$

# Dynamic emulation

* Our simulation model is a time evolving dynamical system
* Often desirable to emulate dynamical behaviour over time steps (much larger than the simulation time steps)
* Regard the simulator as a dynamic function
$$ \mathbf{y}_t = f(\mathbf{x}_t,\mathbf{y}_{t-1}) $$
where $\mathbf{y}_t$ is a state vector and $\mathbf{x}_t$ represents the model inputs at time $t$
* Statistically model and emulate $f(\cdot,\cdot)$

# Emulation results

![Emulation for flocs](figs/olu-009.png){height=60%}

# Emulation results

![Emulation for biofilm](figs/olu-011.png){height=60%}

# Emulation strategy

![Emulation for upscaling](figs/olu-008.png){height=60%}

# Mesoscale modelling

* Not quite ready to model a full-scale WTP or even our pilot plant
* Want to start with modelling experiments with bench-scale experiments within the pilot plant facility (with real wastewater)
* Replicate flow cells consisting of channels approx 2cm wide and 40cm long
* Waste water pulsed through them as slowly as practical for 2 weeks
* Monitor biofilm formation and composition and wastewater composition over time
* Develop a continuum computational model of this experiment

# Emulation as an upscaling strategy

* Discretise continuum model in 2d, covering bottom of flow cell (eg. squares with side 200um)
* Model flow velocity, $\mathbf{u}(\mathbf{x},t)$, water height $H(\mathbf{x},t)$, biofilm height, $h(\mathbf{x},t)$ and composition $\boldsymbol{\theta}(\mathbf{x},t)$, nutrient concentrations, $\boldsymbol{\phi}(\mathbf{x},t)$, etc.
* In order to implement this model, many rates of change and numerical derivatives are required
* eg. the rate of growth of the biofilm given its current composition and (nutrient) environment
* These rates can be provided by an emulator trained on runs from the individual-based model
* Embedding fast approximate/stochastic emulators into the continuum model provides a natural bridge from the fine-scale information provided by the IB model to the next scale up
* Work in progress...

# Acknowledgements 1: Funders and collaborators

![NUFEB](figs/NUFEB){height=20%}
![EPSRC](figs/epsrc){height=30%}

.

.

![Newcastle University](figs/Newcastle_Master_Col){height=10%}
 . . . . . . 
![ORNL](figs/ornl){height=10%} 

.

.

![NWL](figs/NWL){height=10%} . . . . . . .
![MM](figs/mm){height=10%}

# Acknowledgements 2: Newcastle People

## Investigators

* Engineering: **Tom Curtis**, Dana Ofiteru, Jinju Chen, Ben Bridgens, Russel Davenport
* Maths & Stats: DJW, David Swailes
* Computing: Steve McGough, Paolo Zuliani, Anil Wipat
* Biology: Steve Rushton

## Modelling RAs

* Engineering: *Pahala Jayathilake*, Prashant Gupta, Ben Allen, Andrew Coughtree, Rebeca Gonzales-Cabaleiro
* Maths & Stats: *Oluwole Oyebamiji*
* Computing: *Bowen Li*, Curtis Madsen
* Biology: Prashant Gupta

# References

## Papers

* Pahala Gedara, J. *et al.* (2017) A mechanistic individual-based model of microbial communities, *PLoS ONE*, **12**(8):e0181965.
* Oyebamiji, O. *et al.* (2017) Gaussian process emulation of an individual-based model simulation of microbial communities, *Journal of Computational Science*, **22**: 69-84.
* Oyebamiji, O. *et al.* (2017) A surrogate-based approach to modelling the impact of hydrodynamic shear stress on biofilm deformation, in submission.

## Web

* http://research.ncl.ac.uk/nufeb
* http://tinyurl.com/darrenjw

