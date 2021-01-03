#!/usr/bin/env bash
# exit when any command fails
set -e

ROOT=$(realpath "$(dirname $0)/..")
test_stations_path="${ROOT}/data/test_stations.csv"
test_stations_string=$(cat ${test_stations_path} | cut -d ',' -f6 | tail -n +2)
echo ${test_stations_string}

nbatch=32

GPmodel="SExSE"
for ICAO in ${test_stations_string}; do
    echo "ICAO parameter fitting: " $ICAO
    # control the number of simultaneous processes
    # https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop
    ((i++%nbatch==0)) && wait
    sleep $(echo "${RANDOM}/32767/4+2" | bc -l) # sleep for a random time between 0 and 0.25s

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

echo "--- predict from nearby ---"
for ICAO in ${test_stations_string}; do
    # control the number of simultaneous processes
    echo "ICAO predict from nearby: " $ICAO
    ((i++%nbatch==0)) && wait
    sleep $(echo "${RANDOM}/32767/4+2" | bc -l) # sleep for a random time between 0 and 0.25s

    kerneljson="data/outputs/fitted_kernel/mll/${GPmodel}/hyperparams_${GPmodel}_${ICAO}.json"
    dvc run \
        --name nearby_predictions_${ICAO}_${GPmodel} \
        --force \
        --wdir $ROOT \
        --deps batch/pipeline1.jl \
        --deps data/data2015 \
        --deps src/preprocessing.jl \
        --deps src/GPrealisations.jl \
        --deps src/predict_from_nearby.jl \
        --deps ${kerneljson} \
        --outs data/outputs/predictions_from_nearby/mll/${GPmodel}/${ICAO} \
        julia --project=batch batch/pipeline1.jl ${ICAO} ${GPmodel} ./data ./data/outputs ${kerneljson} \
            --knearest=5 \
            & # do in parallel
done

wait

ksmoothmax=50
epsilon=0.01

echo "--- compile Stan ---"
compiledstandir=$(julia --project=batch batch/pipeline2.jl \
    outputdir \
    KBDL \
    SExSE \
    1 \
    ./data \
    ./data/outputs/compiledstan
)
echo "compiled stan directory: " $compiledstandir
dvc run \
    --name compilestan \
    --force \
    --wdir $ROOT \
    --deps batch/pipeline2.jl \
    --deps src/stan_impute.jl \
    --outs "${compiledstandir}/imputation.stan" \
    --outs "${compiledstandir}/imputation.o" \
    --outs "${compiledstandir}/imputation.hpp" \
    --outs "${compiledstandir}/imputation.d" \
    --outs "${compiledstandir}/imputation" \
    julia --project=batch batch/pipeline2.jl \
        compilestan \
        KBDL \
        SExSE \
        1 \
        ./data \
        ./data/outputs/compiledstan

wait 

echo "--- stan imputations ---"
for ICAO in ${test_stations_string}; do
    echo "ICAO stan imputations: " $ICAO
    for windownum in {1..3}; do
        # control the number of simultaneous processes
        outputdir=$(julia --project=batch batch/pipeline2.jl \
            outputdir \
            $ICAO \
            $GPmodel \
            ${windownum} \
            ./data \
            ./data/outputs
            )
        cp -R ${compiledstandir}/* ${outputdir}/
        ((i++%nbatch==0)) && wait
        sleep $(echo "${RANDOM}/32767/4+2" | bc -l) # sleep for a random time between 0 and 0.25s

        dvc run \
            --name imputations_${ICAO}_${GPmodel}_${windownum} \
            --force \
            --wdir $ROOT \
            --deps batch/pipeline2.jl \
            --deps data/data2015 \
            --deps src/preprocessing.jl \
            --deps src/stan_impute.jl \
            --deps data/outputs/predictions_from_nearby/mll/${GPmodel}/${ICAO} \
            --deps batch/batch_chunks.jl \
            --deps "${compiledstandir}/imputation.stan" \
            --deps "${compiledstandir}/imputation.o" \
            --deps "${compiledstandir}/imputation.hpp" \
            --deps "${compiledstandir}/imputation.d" \
            --deps "${compiledstandir}/imputation" \
            --outs "${outputdir}/imputation.stan" \
            --outs "${outputdir}/imputation.o" \
            --outs "${outputdir}/imputation.hpp" \
            --outs "${outputdir}/imputation.d" \
            --outs "${outputdir}/imputation_1.data.R" \
            --outs "${outputdir}/imputation_samples_1.csv" \
            --outs "${outputdir}/imputation_samples_2.csv" \
            --outs "${outputdir}/imputation_samples_3.csv" \
            --outs "${outputdir}/imputation_samples_4.csv" \
            "
                julia --project=batch batch/pipeline2.jl \
                    impute \
                    $ICAO \
                    $GPmodel \
                    ${windownum} \
                    ./data \
                    ./data/outputs \
                    --seed=${windownum} \
                    --ksmoothmax=$ksmoothmax \
                    --epsilon=$epsilon;
                # remove last 4 lines of Stan output, which contains timings
                # and is therefore not reproducible
                grep -v ' seconds' \"${outputdir}/imputation_samples_1.csv\" > \"${outputdir}/tmp.csv\" \
                    && mv \"${outputdir}/tmp.csv\" \"${outputdir}/imputation_samples_1.csv\";
                grep -v ' seconds' \"${outputdir}/imputation_samples_2.csv\" > \"${outputdir}/tmp.csv\" \
                    && mv \"${outputdir}/tmp.csv\" \"${outputdir}/imputation_samples_2.csv\";
                grep -v ' seconds' \"${outputdir}/imputation_samples_3.csv\" > \"${outputdir}/tmp.csv\" \
                    && mv \"${outputdir}/tmp.csv\" \"${outputdir}/imputation_samples_3.csv\";
                grep -v ' seconds' \"${outputdir}/imputation_samples_4.csv\" > \"${outputdir}/tmp.csv\" \
                    && mv \"${outputdir}/tmp.csv\" \"${outputdir}/imputation_samples_4.csv\";

                rm \"${outputdir}/imputation_2.data.R\";
                rm \"${outputdir}/imputation_3.data.R\";
                rm \"${outputdir}/imputation_4.data.R\";
            " &
    done
done

wait
