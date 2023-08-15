# Environmental Crisis Information Diffusion Model

## Nicholas Rabb, Catherine Knox, Nitya Nadgir, Shafiqul Islam

### Corresponding author: Nicholas Rabb (nicholas.rabb@tufts.edu)

## Table of Contents

* [Model Description](#model-description)
* [Replication Instructions](#replication-instructions)

## Model Description

## Replication Instructions

### Prerequisites

Please make sure that you have the following software installed:
* NetLogo 6.1.1
* Python 3 or higher

And the following Python packages installed in the environment you are using:

* numpy
* matplotlib
* scipy
* pandas
* networkx
* sklearn
* websockets
* python-louvain

### Replication with NetLogo and BehaviorSpace

Open the `cognitive-contagion.nlogo` file in NetLogo 6.1.1. Open BehaviorSpace by navigating to *Tools --> BehaviorSpace*. Here, you can select any of the experiments to run by clicking on its name, clicking *Run*, unchecking all the checkboxes on the *Run Options* dialogue, setting the number of cores you would like it to run on, then clicking *Ok*.

In Rabb et al., 2023, several of the BehaviorSpace experiments supported the results presented:
* static-influence-sweep
* base-model-sweep
* static-influence-monte-carlo-[1-5]_no-organizing
* static-influence-monte-carlo-EP-[1-5]_no-organizing
* programmed-influence-monte-carlo
* programmed-influence-monte-carlo_random-graphs

Our results from running these simulations and aggregating data into a `pandas` dataframe can be found in the `data` directory. In the same order as the above experiment names, below are the locations of each experiment's results:

* `data/influence-model-sweep.csv`
* `data/base-model-sweep.csv`
* `data/monte-carlo-flint/monte-carlo-simulations/monte-carlo-[1-5]_no-organizing.csv`
* `data/monte-carlo-ep/monte-carlo-ep-[1-5]_no-organizing.csv`
* TODO: Fix this `data/programmed_flint_peaks/monte-carlo_one-graph.csv`

Loading these files into `pandas` using `pd.read_csv(<your_file_name>)` will allow analysis of all simulation results. The simulation output data is of the following format:

```
index | n | spread-type | simple-spread-chance | graph-type | ba-m | citizen-citizen-influence | citizen-media-influence | repetition | data | num_media
```

* `index`: The index of the data frame
* `n`: The number of citizen agents in the graph
* `spread-type`: Always *simple*
* `simple-spread-chance`: The *p* parameter from the independent cascade model
* `graph-type`: Always *barabasi-albert*
* `ba-m`: The *m* parameter from the graph formation process
* `citizen-citizen-influence`: The *p_cc* parameter from the heterogeneous independent cascade model (set to 1 in the base model)
* `citizen-media-influence`: The *p_cm* parameter from the heterogeneous independent cascade model (set to 1 in the base model)
* `repetition`: Which random graph (out of 5 total per parameter combination) is being run for this simulation trial
* `data`: The time series data of how many agents newly adopted the environmental crisis belief at a given time step
* `num_media`: The number of media agents in this simulation

The Monte Carlo simulation data has an extra column `run_id`, which corresponds to the Monte Carlo simulation number.