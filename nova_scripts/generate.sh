#!/bin/bash

export SRT_PRIVATE_CONTEXT=$PWD
export SRT_BASE_RELEASE="development"
setup_fnal_security

seed_list=(460 555 684 201 541 91 192 448 606 841 986 45 489 339 161 25)

for i in "${seed_list[@]}"
do
    echo "Seed : $i"

    mkdir -p "/pnfs/nova/scratch/users/karthiva/NOvA_CNF/global_fits/$i"

    ./NuXAna/nus5p1/grid/nus5p1_submit \
        --njobs 1 \
        --tarball Generate_Preds.tar.bz2 \
        --executable nus5p1_global_fit \
        --args th24vsdm41 numusel_ncres30sel_nuonesel_fhc_rhc_neardet_fardet "$i" all verbose \
        --outdir "/pnfs/nova/scratch/users/karthiva/NOvA_CNF/global_fits/$i" \
        --lifetime=6h \
        -d 0.5GB \
        -m 0.5GB

    echo "Submission $i successful. Waiting 30 seconds."

    sleep 30
done
