#!/bin/bash

# Define source, target, maxdepth and cd to source
GPmodel="$1"
source="/Volumes/Samsung250GB/temperature_model/saved/hr_measure/$GPmodel"
target="ody:/n/regal/pillai_lab/mrischard/temperature_model/saved/hr_measure/$GPmodel"
depth=1
cd "${source}"

# Set the maximum number of concurrent rsync threads
maxthreads=5
# How long to wait before checking the number of rsync threads again
sleeptime=10

# Find all folders in the source directory within the maxdepth level
find . -maxdepth ${depth} -type d | while read dir
do
    # Make sure to ignore the parent folder
    if [ `echo "${dir}" | awk -F'/' '{print NF}'` -gt ${depth} ]
    then
        # Strip leading dot slash
        subfolder=$(echo "${dir}" | sed 's@^\./@@g')
		echo "${subfolder}"
        # if [ ! -d "${target}/${subfolder}" ]
        # then
        #     # Create destination folder and set ownership and permissions to match source
        #     mkdir -p "${target}/${subfolder}"
        #     chown --reference="${source}/${subfolder}" "${target}/${subfolder}"
        #     chmod --reference="${source}/${subfolder}" "${target}/${subfolder}"
        # fi
        # Make sure the number of rsync threads running is below the threshold
        while [ `ps -ef | grep -c "[r]sync --verbose"` -gt ${maxthreads} ]
        do
            echo "Sleeping ${sleeptime} seconds"
            sleep ${sleeptime}
        done
        # Run rsync in background for the current subfolder and move on to the next one
        # nohup rsync --verbose --human-readable --progress --archive --compress  --update "${target}/${subfolder}/" "${source}/${subfolder}/" </dev/null >/dev/null 2>&1 &
        rsync --verbose --human-readable --progress --archive --compress  --update "${target}/${subfolder}/" "${source}/${subfolder}/" &
    fi
done
while [ `ps -ef | grep -c "[r]sync --verbose" -gt 0 ]
do
    echo "Sleeping 30 seconds; waiting for all rsync jobs to finish"
    sleep 30
done
