# Fuzzy-TCAP-Code-and-Experiments
This repository contains the implementation of Fuzzy TCAP, an extension of the Targeted Correct Attribution Probability (TCAP) metric for assessing disclosure risk in synthetic microdata. The core Python script replicates the original TCAP logic when all parameters are set as follows: all variable weights = 1, tau (τ) = 1, and both minimum and maximum key depth = 6–6.

The project introduces two methodological extensions:

Fuzzy key selection, which relaxes strict matching assumptions by evaluating variable-length key combinations through a lattice of nested key subsets.

Variable weighting, which models intruder strategies based on either (a) external data availability or (b) statistical predictiveness of key variables.

A separate script is provided to compute weights using Cramér’s V, capturing the predictive power of each key variable on the target. These scores are normalised to sum to 1.

The repository also includes a Jupyter notebook for synthesising the Canada 2011 census dataset using Synthpop’s CART method, with options to subset key and target variables. This code was used to prepare synthetic data for Fiji (2007), Rwanda (2012), and UK (1991) as well.

In total, 144 experiments were conducted across 3 target variables, 4 depth levels, and 3 weighting strategies in 4 datasets. All outputs, including per-variable risk scores and merged scores for visualisation, are included as Jupyter notebooks in this repository.
