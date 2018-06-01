This repository contains code and jupyter notebooks for the paper “Bias correction in daily maximum and minimum temperature measurements through Gaussian process modeling” available on [arxiv.org](https://arxiv.org/abs/1805.10214).
The code and notebooks are (almost all) written in the [julia](https://julialang.org) programming language.

# List of Figures

The figures in the manuscript are generated in notebooks, so they are fully reproducible.
The table below indicates which notebook each figure was generated in.

| #  | File Name             | Notebook                            | Short Caption                                                                                                                                         |
|----|-----------------------|-------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | waterloo_triangles    | Waterloo_bias.ipynb                 | An extract of the temperature measurements from KALO showing bias due to measurement hour.                                                            |
| 2  | Iowa_map              | JuliaGP_spatial_variogram.ipynb     | Map of the four airport weather stations in Iowa providing hourly temperature records.                                                                |
| 3  | waterloo_avgTnTx      | Waterloo_bias.ipynb                 | Mean daily Tx (top left) and Tn (top right), and mean absolute daily change in Tx (bottom left) and Tn (bottom right).                                |
| 4  | imputations_2x2       | BatchDiagnostics.ipynb              | Imputations of the temperature time series at Waterloo Municipal Airport (KALO) between May 28, 2015 and June 1, 2015.                                |
| 5  | constraints3d         | (drawn in a vector graphics editor) | With three variables X_1, and X_2 and X_3, F_{X|Xmin,Xmax} resides in the one-dimensional six-sided loop shown with thicker green lines.              |
| 6  | toy_quantiles         | SmoothMax Simulation.ipynb          | Marginal distribution of F_X and F_{X|Xmin,Xmax}.                                                                                                     |
| 7  | toy_joint             | SmoothMax Simulation.ipynb          | Comparison of the joint joint PDF of X_23 and X_52 obtained analytically and from SmoothHMC samples.                                                  |
| 8  | spatial_variogram     | JuliaGP_spatial_variogram.ipynb     | Semi-variograms of the temperature temperature time series at four Iowa weather stations.                                                             |
| 9  | imputed_summary_stats | ImputedSummaryStatistics.ipynb      | Inference from imputations of summary statistics (average Tn and average Tx).                                                                         |
| 10 | measure_hour_example  | BatchMeasurementHour.ipynb          | Constrained and unconstrained imputations in an eight-day window, assuming (top) the correct measurement hour, and (bottom) a wrong measurement hour. |
| 11 | hr_inference          | BatchMeasurementHour.ipynb          | Concordance for imputations of temperatures at KALO assuming measurement hour=1,…,24.                                                                 |

# temperature_model

For each model, there is a notebook for optimization and a notebook for imputations. 
Here's a table of contents of sorts:

| Model name   | Optimization     | Imputation          | Description                                        |
|--------------|------------------|---------------------|----------------------------------------------------|
| SExSE        | JuliaGP_spatial2 | Julia_toStan4       | Simple product of squared exponentials             |
| SESE_diurnal | JuliaGP_spatial2 | Julia_toStan5       | add diurnal component (with its own diurnal decay) |
| sumprod      | JuliaGP_spatial4 | Julia_toStan6.ipynb | Most flexible model, all parameters optimized      |
| free_var     | JuliaGP_spatial3 | Julia_toStan3.ipynb | Same as fixed_var, but free variance parameters    |
| fixed_var    | JuliaGP_spatial  | Julia_toStan2       | Sum of products, but with variance fixed           |

