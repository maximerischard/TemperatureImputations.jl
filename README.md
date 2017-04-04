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
