#!/usr/bin/env bash

ROOT=$(grealpath "$(dirname $0)/..")
test_stations_path="${ROOT}/data/test_stations.csv"
test_stations_string=$(cat ${test_stations_path} | cut -d ',' -f6 | tail -n +2 | head -n 2)
echo ${test_stations_string}

nbatch=2

GPmodel="SExSE"
for ICAO in ${test_stations_string}; do
    # control the number of simultaneous processes
    # https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
    (i=i%nbatch) ; ((i++==0)) && wait

    echo "ICAO: " $ICAO
# echo $test_stations | xargs -I {} \
    dvc run \
        --name kernelfit_${ICAO}_${GPmodel} \
        --force \
        --wdir $ROOT \
        --deps batch/pipeline0_kernelfit.jl \
        --deps data/data2015 \
        --deps src/preprocessing.jl \
        --outs-no-cache data/outputs/fitted_kernel/mll/${GPmodel}/hyperparams_${GPmodel}_${ICAO}.json \
        julia --project=batch batch/pipeline0_kernelfit.jl ${ICAO} ${GPmodel} \
            ./data ./data/outputs --timelimit=Inf --knearest=5 \
            & # do in parallel

done

wait

for ICAO in ${test_stations_string}; do
    # control the number of simultaneous processes
    (i=i%nbatch) ; ((i++==0)) && wait
    dvc run \
        --name nearby_predictions_${ICAO}_${GPmodel} \
        --force \
        --wdir $ROOT \
        --deps batch/pipeline1.jl \
        --deps data/data2015 \
        --deps src/preprocessing.jl \
        --deps src/GPrealisations.jl \
        --deps src/predict_from_nearby.jl \
        --deps data/outputs/fitted_kernel/mll/${GPmodel}/hyperparams_${GPmodel}_${ICAO}.json \
        --outs data/outputs/predictions_from_nearby/mll/${GPmodel}/${ICAO} \
        julia --project=batch batch/pipeline1.jl ${ICAO} ${GPmodel} ./data ./data/outputs \
            --knearest=5 \
            & # do in parallel
done

ksmoothmax=50
epsilon=0.01

wait

compiledstandir=$(julia --project=batch batch/pipeline2.jl \
    outputdir \
    KBDL \
    SExSE \
    1 \
    ./data \
    ./data/outputs/compiledstan \
    --seed=42 \
    --ksmoothmax=Inf \
    --epsilon=Inf
)
dvc run \
    --name compilestan \
    --force \
    --wdir $ROOT \
    --deps batch/pipeline2.jl \
    --deps src/stan_impute.jl \
    --outs ${compiledstandir} \
    julia --project=batch batch/pipeline2.jl \
        compilestan \
        KBDL \
        SExSE \
        1 \
        ./data \
        ./data/outputs/compiledstan \
        --seed=42 \
        --ksmoothmax=Inf \
        --epsilon=Inf

wait 
for ICAO in ${test_stations_string}; do
    for windownum in {1..3}; do
        # control the number of simultaneous processes
        outputdir=$(julia --project=batch batch/pipeline2.jl \
            outputdir \
            $ICAO \
            $GPmodel \
            ${windownum} \
            ./data \
            ./data/outputs \
            --seed=${windownum} \
            # --crossval \
            --ksmoothmax=$ksmoothmax \
            --epsilon=$epsilon)
        cp ${compiledstandir}/* ${outputdir}/*
        (i=i%nbatch) ; ((i++==0)) && wait
        dvc run \
            --name imputations_${ICAO}_${GPmodel}_${windownum} \
            --force \
            --wdir $ROOT \
            --deps batch/pipeline2.jl \
            --deps data/data2015 \
            --deps src/preprocessing.jl \
            --deps src/stan_impute.jl \
            --deps data/outputs/predictions_from_nearby/mll/${GPmodel}/${ICAO} \
            --deps $compiledstandir \
            --outs ${outputdir} \
            julia --project=batch batch/pipeline2.jl \
                impute \
                $ICAO \
                $GPmodel \
                ${windownum} \
                ./data \
                ./data/outputs \
                --seed=${windownum} \
                # --crossval \
                --ksmoothmax=$ksmoothmax \
                --epsilon=$epsilon &
    done
done

wait
