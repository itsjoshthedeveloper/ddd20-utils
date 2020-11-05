#!/bin/bash
set -e
# Set these for you:
OUT_DIR=processed_dataset
ORIGIN_DIR=data/fordfocus/jun25
# TODO_FILES=( run5/rec1487839456.hdf5 run5/rec1487844247.hdf5 run5/rec1487849151.hdf5 run5/rec1487856408.hdf5 run5/rec1487858093.hdf5 run5/rec1487842276.hdf5 run5/rec1487846842.hdf5 run5/rec1487849663.hdf5 run5/rec1487857941.hdf5 run5/rec1487860613.hdf5 run5/rec1487864316.hdf5 )
TODO_FILES=( rec1498343773.hdf5  rec1498355604.hdf5  rec1498402981.hdf5  rec1498410825.hdf5 rec1498348424.hdf5  rec1498392691.hdf5  rec1498410237.hdf5  rec1498411982.hdf5  rec1498412736.hdf5 )
# TODO_FILES=( rec1498411982.hdf5 )
# TODO_FILES=( rec1498410237.hdf5 )
# Here on down should not require modification
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
    # ------------ Prepare APS -------------#
    ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_frames_100ms.hdf5 --rewrite 1 --skip_mean_std 1
    # ----------- Prepare timestep split DVS ------- #
    ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_bin100ms_with_timesteps.hdf5 --rewrite 1 --skip_mean_std 1 --split_timesteps --timesteps 20
    # ----------- Prepare accumulated DVS ----------- #
    ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_bin100ms_dvs_accum_frames.hdf5 --rewrite 1 --skip_mean_std 1

    # ----------- Prepare Encoder Decoder Dataset ----------- #
    ipython ./prepare_simul_cnn_data.py -- --filename_aps ${OUT_FULL_FILE_PREFIX}_frames_100ms.hdf5 --filename_dvs ${OUT_FULL_FILE_PREFIX}_bin100ms_with_timesteps.hdf5 --rewrite 1 --skip_mean_std 1 --split_timesteps --timesteps 20
done

# ------------------- Find all APS datasets ---------------- #
for filename in ${OUT_DIR}/*_frames_100ms.hdf5
do
    frames_h5list="$frames_h5list $filename"
    frames_type_list="$frames_type_list aps_frame_80x80"
    echo "### Found the following APS datasets: ${frames_h5list} ###"
done
# ------------------- Find all accumulated DVS datasets --------- #
for filename in ${OUT_DIR}/*_bin100ms_dvs_accum_frames.hdf5
do
    dvs_accum_frames_h5list="$dvs_accum_frames_h5list $filename"
    dvs_accum_frames_type_list="$dvs_accum_frames_type_list dvs_frame_80x80"
    echo "### Found the following DVS accumulated frames datasets: ${dvs_accum_frames_h5list} ###"
done
# ------------------ Find all timestep seperated DVS datasets ------ #
for filename in ${OUT_DIR}/*_bin100ms_with_timesteps.hdf5
do
    dvs100ms_h5list="$dvs100ms_h5list $filename"
    dvs100ms_type_list="$dvs100ms_type_list dvs_frame_80x80"
    echo "### Found the following constant time datasets: ${dvs100ms_h5list} ###"
done

# ------------------ Train Encoder APS to DVS ----------- #
ipython ./multitrain_test_cnn_pytorch.py -- --encoder_decoder --h5files_aps ${frames_h5list[@]} --h5files_dvs ${dvs100ms_h5list[@]} --dataset_keys_aps ${frames_type_list[@]} --dataset_keys_dvs ${dvs100ms_type_list[@]} --run_id encoder_decoder --snn --BNTT --dvs --seperate_dvs_channels --split_timesteps --timesteps 20 --optimizer "SGD"
for TODO_FILE in "${TODO_FILES[@]}"
do
    IN_FULL_FILE_PREFIX=${ORIGIN_DIR}/${TODO_FILE%.*}
    BASE_ID=`basename ${IN_FULL_FILE_PREFIX}`
    OUT_FULL_FILE_PREFIX=${OUT_DIR}/${BASE_ID}
    echo "### Working on $OUT_FULL_FILE_PREFIX ####"

    # ----------- Cache encoded Dataset ------------- #
    ipython ./prepare_encoded_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_bin100ms_with_timesteps.hdf5 --rewrite 1 --timesteps 20
done

# -------------------- Find all encoded frame datasets -------------- #
for filename in ${OUT_DIR}/*_bin100ms_with_timesteps.hdf5
do
    dvs_encoded_h5list="$dvs_encoded_h5list $filename"
    dvs_encoded_type_list="$dvs_encoded_type_list encoded_frame_80x80"
    echo "### Found the following constant time datasets: ${dvs_encoded_h5list} ###"
done

# Train the networks

# ------------------ ANN on APS -------------- #
ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${frames_h5list[@]} --dataset_keys ${frames_type_list[@]} --run_id ann_only_aps --optimizer "Adam"

# ------------------ ANN on accumulated DVS -----------#
ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${dvs_accum_frames_h5list[@]} --dataset_keys ${dvs_accum_frames_type_list[@]} --run_id ann_only_acc_dvs --optimizer "Adam"

# ------------------ ANN on APS + Accumulated DVS ---------- #
ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${frames_h5list[@]} ${dvs_accum_frames_h5list[@]} --dataset_keys ${frames_type_list[@]} ${dvs_accum_frames_type_list[@]} --run_id ann_combined_aps_and_acc_dvs --optimizer "Adam"

# ------------------ ANN on encoded DVS -------#
ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${dvs_encoded_h5list[@]} --dataset_keys ${dvs_encoded_type_list[@]} --run_id ann_only_encoded_dvs --optimizer "Adam"

# ------------------ ANN on APS + Encoded DVS -------------#
ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${frames_h5list[@]} ${dvs_encoded_h5list[@]} --dataset_keys ${frames_type_list[@]} ${dvs_encoded_type_list[@]} --run_id ann_combined_aps_and_encoded_dvs --optimizer "Adam"

# ------------------ SNN on APS -------------- #
# ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${frames_h5list[@]} --dataset_keys ${frames_type_list[@]} --run_id aps_snn --snn --BNTT --timesteps 20 --optimizer "SGD"

# ------------------ SNN on timestep split DVS ------------#
# ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${dvs100ms_h5list[@]} --dataset_keys ${dvs100ms_type_list[@]} --run_id dvs_snn --dvs --snn --BNTT --split_timesteps --seperate_dvs_channels --timesteps 20 --optimizer "SGD"

# ------------------ SNN on rate coded DVS --------------- #
# ipython ./multitrain_test_cnn_pytorch.py -- --h5file ${dvs100ms_h5list[@]} --dataset_keys ${dvs100ms_type_list[@]} --run_id dvs100ms_multi --dvs --seperate_dvs_channels --snn

# ------------------ ANN + Encoder (With backprop on encoder) ------- #
# ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${dvs100ms_h5list[@]} --dataset_keys ${dvs100ms_type_list[@]} --run_id ann_back_prop_encoder_only_dvs --dvs --use_encoder --split_timesteps --seperate_dvs_channels --timesteps 20 --optimizer "Adam"

# ------------------- Evaluate the system ------ #
ipython ./evaluate.py -- --h5files_aps ${frames_h5list[@]}\
                         --dataset_keys_aps ${frames_type_list[@]}\
                         --h5files_dvs_frames ${dvs_accum_frames_h5list[@]}\
                         --dataset_keys_dvs_frames ${dvs_accum_frames_type_list[@]}\
                         --h5files_dvs_timesteps ${dvs100ms_h5list[@]}\
                         --dataset_keys_dvs_timesteps ${dvs100ms_type_list[@]}\
                         --h5files_combined_frames ${frames_h5list[@]} ${dvs_accum_frames_h5list[@]}\
                         --dataset_keys_combined_frames ${frames_type_list[@]} ${dvs_accum_frames_type_list[@]}\
                         --h5files_dvs_encoded ${dvs_encoded_h5list[@]}\
                         --dataset_keys_dvs_encoded ${dvs_encoded_type_list[@]}\
                         --h5files_combined_aps_dvs_enc_frames ${frames_h5list[@]} ${dvs_encoded_h5list[@]}\
                         --dataset_keys_combined_aps_dvs_enc_frames ${frames_type_list[@]} ${dvs_enc_frames_type_list[@]}\
                         --timesteps 20\
                         --model_dir "./saved_models"

