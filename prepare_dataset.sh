#!/bin/bash
set -e
# Set these for you:
OUT_DIR=processed_dataset_full/night
ORIGIN_DIR=data/fordfocus/
# TODO_FILES=( aug12/rec1502599151.hdf5 ) # Tiny data - one clip from night one from day
TODO_FILES=( jul09/rec1499656391.hdf5 jul09/rec1499657850.hdf5 aug01/rec1501649676.hdf5 aug01/rec1501650719.hdf5 aug05/rec1501994881.hdf5 aug09/rec1502336427.hdf5 aug09/rec1502337436.hdf5 jul01/rec1498946027.hdf5 aug01/rec1501651162.hdf5 jul02/rec1499025222.hdf5 aug09/rec1502338023.hdf5 aug09/rec1502338983.hdf5 aug09/rec1502339743.hdf5 jul01/rec1498949617.hdf5 aug12/rec1502599151.hdf5 ) # Night data
# TODO_FILES=( jul16/rec1500220388.hdf5 jul18/rec1500383971.hdf5 jul18/rec1500402142.hdf5 jul28/rec1501288723.hdf5 jul29/rec1501349894.hdf5 aug01/rec1501614399.hdf5 aug08/rec1502241196.hdf5 aug15/rec1502825681.hdf5 jul02/rec1499023756.hdf5 jul05/rec1499275182.hdf5 jul08/rec1499533882.hdf5 jul16/rec1500215505.hdf5 jul17/rec1500314184.hdf5 jul17/rec1500329649.hdf5 aug05/rec1501953155.hdf5 ) # Day data
# TODO_FILES=( jul09/rec1499656391.hdf5 jul09/rec1499657850.hdf5 aug01/rec1501649676.hdf5 aug01/rec1501650719.hdf5 aug05/rec1501994881.hdf5 aug09/rec1502336427.hdf5 aug09/rec1502337436.hdf5 jul01/rec1498946027.hdf5 aug01/rec1501651162.hdf5 jul02/rec1499025222.hdf5 aug09/rec1502338023.hdf5 aug09/rec1502338983.hdf5 aug09/rec1502339743.hdf5 jul01/rec1498949617.hdf5 aug12/rec1502599151.hdf5 jul16/rec1500220388.hdf5 jul18/rec1500383971.hdf5 jul18/rec1500402142.hdf5 jul28/rec1501288723.hdf5 jul29/rec1501349894.hdf5 aug01/rec1501614399.hdf5 aug08/rec1502241196.hdf5 aug15/rec1502825681.hdf5 jul02/rec1499023756.hdf5 jul05/rec1499275182.hdf5 jul08/rec1499533882.hdf5 jul16/rec1500215505.hdf5 jul17/rec1500314184.hdf5 jul17/rec1500329649.hdf5 aug05/rec1501953155.hdf5 ) # Day + Night data

# --------------------------- Preprocess Data --------------------------- #
# <<'###BLOCK-COMMENT'
for TODO_FILE in "${TODO_FILES[@]}"
do
    IN_FULL_FILE_PREFIX=${ORIGIN_DIR}/${TODO_FILE%.*}
    BASE_ID=`basename ${IN_FULL_FILE_PREFIX}`
    OUT_FULL_FILE_PREFIX=${OUT_DIR}/${BASE_ID}
    echo "### Working on $OUT_FULL_FILE_PREFIX ####"

    # Export data
    # ------------- Export APS ----------- #
    ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize 0.100 --export_aps 1 --export_dvs 0 --out_file ${OUT_FULL_FILE_PREFIX}_frames_100ms.hdf5

    # ------------- Export timestep seperated DVS ------- #
    ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize 0.100 --export_aps 0 --export_dvs 1 --out_file ${OUT_FULL_FILE_PREFIX}_bin100ms_with_timesteps.hdf5 --split_timesteps --timesteps 20

    # ------------- Export accumulated DVS -------------#
    ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize 0.100 --export_aps 0 --export_dvs 1 --out_file ${OUT_FULL_FILE_PREFIX}_bin100ms_dvs_accum_frames.hdf5

    # Prepare and resize
    # ----------- Prepare Encoder Decoder Dataset ----------- #
    ipython ./prepare_simul_cnn_data.py -- --filename_aps ${OUT_FULL_FILE_PREFIX}_frames_100ms.hdf5 --filename_dvs_split ${OUT_FULL_FILE_PREFIX}_bin100ms_with_timesteps.hdf5 --filename_dvs_accum ${OUT_FULL_FILE_PREFIX}_bin100ms_dvs_accum_frames.hdf5 --rewrite 1 --skip_mean_std 1 --split_timesteps --timesteps 20
done
###BLOCK-COMMENT
