using Base.Dates: Day, Hour
njob_str, increment_str, hr_str = ARGS
inc_days = Day(parse(Int, increment_str))
start_date = Date(2015,1,1) + (parse(Int, njob_str)-1)*inc_days
hr_measure = Hour(parse(Int, hr_str))

using Stan
using DataFrames
using GaussianProcesses
using Proj4
using PDMats: PDMat
using DataFrames: head
using JLD
using GaussianProcesses: SumKernel

include("../src/utils.jl")
include("../src/preprocessing.jl")
include("../src/variogram.jl")

isdList=read_isdList(data_dir="..")
isdSubset=isdList[[(usaf in (725450,725460,725480,725485)) for usaf in isdList[:USAF].values],:]
isdSubset

hourly_cat=read_Stations(isdSubset, data_dir="..")
itest=3

TnTx = test_data(hourly_cat, itest, Hour(17))

module pred
    using PDMats: PDMat
    using DataFrames
    using GaussianProcesses: GP, Kernel, MeanZero, predict
    using Base.Dates: Day, Hour
    using Stan
    using DataFrames: DataFrame, by

    include("../src/utils.jl")
    include("../src/predict_from_nearby.jl")
    include("../src/stan_impute.jl")
end

nearby_pred=load("../saved/predictions_from_nearby/725480_2015-01-01_to_2015-03-14.jld")["nearby_pred"]

imputation_data=pred.prep_data(nearby_pred, TnTx, Date(2015,3,1), Hour(17), Day(5))

imputation_model = pred.get_imputation_model();


imputation_data=pred.prep_data(nearby_pred, TnTx, start_date, hr_measure, inc_days*3)
@time sim1 = stan(
    imputation_model, 
    [imputation_data], 
    CmdStanDir=Stan.CMDSTAN_HOME, 
    summary=false, 
    diagnostics=false
    )

save(@sprintf("../saved/chains/imputed_%s_n%s_%s_%shrs.jld", 
    start_date, njob_str, increment_str, hr_str),
    "sim",
    sim1)
