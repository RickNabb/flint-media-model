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

### Data Replication with NetLogo and BehaviorSpace

Open the `cognitive-contagion.nlogo` file in NetLogo 6.1.1. Open BehaviorSpace by navigating to *Tools --> BehaviorSpace*. Here, you can select any of the experiments to run by clicking on its name, clicking *Run*, unchecking all the checkboxes on the *Run Options* dialogue, setting the number of cores you would like it to run on, then clicking *Ok*.

In Rabb et al., 2023, several of the BehaviorSpace experiments supported the results presented:
* static-influence-sweep
* base-model-sweep
* static-influence-monte-carlo-flint-[1-5]_no-organizing
* static-influence-monte-carlo-EP-[1-5]_no-organizing
* programmed-influence-monte-carlo
* programmed-influence-monte-carlo_random-graphs

Our results from running these simulations and aggregating data into a `pandas` dataframe can be found in the `data` directory. In the same order as the above experiment names, below are the locations of each experiment's results:

* `data/static-influence-sweep.csv`
* `data/base-model-sweep.csv`
* `data/f_mc[1-5]_with_metrics.csv`
* `data/ep_mc[1-5]_with_metrics.csv`
* `data/flint_dyn_one_graph_with_metrics.csv`
* `data/flint_dyn_random_graphs_with_metrics.csv`

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

The Monte Carlo simulation data has some extra columns -- `run_id`, `short list`, `unscaled-list`, `threshold`, `peak-time`, `total-spread` -- which correspond to the following:

* `run_id`: The unique ID of the run out of 1000
* `short list`: 
* `unscaled-list`: A copy of the data column
* `threshold`: ???
* `peak-time`: The index of the maximum value in `data`
* `total-spread`: How many agents total the belief spread to throughout the simulation

If you want to generate `pandas` dataframes from your own custom running of simulations (as described above), you can use the interactive Python terminal to do so. After your results from the simulations have been output to some directory, you can load the data processing functions by running `from data import *`. Then, the following functions will generate dataframes from your simulation results -- listed in the order of simulations listed above:

* `static_influence_sweep_results_to_df(<your_sim_output_path>/static-influence-sweep/)`
* `base_model_sweep_results_to_df(<your_sim_output_path>/base-model-sweep/)`
* `static_influence_monte_carlo_results_to_df(<your_sim_output_path>/)`
* `load_static_monte_carlo_dfs('./data/<custom_dir>', '<your_sim_output_path>/flint-monte-carlo/', 5, FLINT_MC_SIM_PARAMS)`
* `load_static_monte_carlo_dfs('./data/<custom_dir>', '<your_sim_output_path>/ep-monte-carlo/', 5, EP_MC_SIM_PARAMS)`
* `programmed_influence_monte_carlo_to_df(<your_sim_output_path>/programmed-influence-monte-carlo/)`
* `programmed_influence_monte_carlo_to_df(<your_sim_output_path>/programmed-influence-monte-carlo_random-graphs/)`

### Analysis Replication

Initial analysis uses the Jupyter Notebook `Data Analysis- Media Agents-Numbered`.  The google trends data is read in and processed with the code under the heading “Google Trends Data Interpretation.” The csv files from both the heterogeneous and base model (`base-model-sweep.csv` and `static-influence-sweep.csv`) are read in to generate a file with added metrics to investigate, including the data used for our time of peak spread analysis.

#### Peak Times for Base & Heterogeneous Models

The Jupyter Notebook for `Base_mod_analysis` is used to generate the histograms of runs that passed our threshold process. The cells first read in the csv files generated from the process above with the metrics. These are then processed to count the total number of simulations in each parameter combination that pass our threshold process, and a column of whether or not 70% of the simulations reach the metric within that parameter combination. Continuing down the notebook, a new csv file is generated that includes a column if the threshold process was successful for the chosen parameters or not. 

By changing the csv file used initially, we can repeat this process for each model version (base or heterogeneous). The last two cells uses the csv files generated above to limit the simulations included to those that passed the threshold and then to create the figure demonstrating histograms of both model versions, with colors corresponding to the base probability of spread, p.

#### Monte Carlo Analyses

The Monte Carlo analysis is completed using the code in the Jupyter Notebook `MC_analysis.` Each of the outputs for the 10 MC results is processed through the `Data Analysis- Media Agents-Numbered` Slide.  The cell labeled “Number of Alignment” could be used to determine the number of simulations that captured the behavior of either Flint or East Palestine by changing the time frame we search for.  In the case of East Palestine, this means that the time of peak were between 0 and 8 (EP peaked at t=3). For Flint, this means limiting the dataframe to the simulations peaking between t=86 and t=96 (within 5 weeks of the peak, occurring at t=91). 

Running the code under “Generate MC Figure” using the data files generated with the metrics in the process explained above (in our folder, these are labeled as `f_mcX_with_metrics.csv` or `ep_mcX_with_metrics.csv`) will generate the figure of all the Monte Carlo Analysis. 

## Google Trends Data

Data capturing Google searches for the Flint Water Crisis and East Palestine train derailment is located in `data/google-trends.csv` and is credited to Google Trends.

## MediaCloud Data

Data capturing media coverage of each crisis is included in the `data/media-cloud` directory, credited to Media Cloud.