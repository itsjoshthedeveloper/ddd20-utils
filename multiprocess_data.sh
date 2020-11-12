#!/bin/bash
set -e
# Set these for you:
OUT_DIR=processed_dataset_test/tiny
RESULT_DIR=saved_models_test/tiny
ORIGIN_DIR=data/fordfocus/
TODO_FILES=( aug12/rec1502599151.hdf5 ) # Tiny data - one clip from night one from day
# TODO_FILES=( jul09/rec1499656391.hdf5 jul09/rec1499657850.hdf5 aug01/rec1501649676.hdf5 aug01/rec1501650719.hdf5 aug05/rec1501994881.hdf5 aug09/rec1502336427.hdf5 aug09/rec1502337436.hdf5 jul01/rec1498946027.hdf5 aug01/rec1501651162.hdf5 jul02/rec1499025222.hdf5 aug09/rec1502338023.hdf5 aug09/rec1502338983.hdf5 aug09/rec1502339743.hdf5 jul01/rec1498949617.hdf5 aug12/rec1502599151.hdf5 ) # Night data
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
    # ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize 0.100 --export_aps 1 --export_dvs 0 --out_file ${OUT_FULL_FILE_PREFIX}_frames_100ms.hdf5

    # ------------- Export timestep seperated DVS ------- #
    # ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize 0.100 --export_aps 0 --export_dvs 1 --out_file ${OUT_FULL_FILE_PREFIX}_bin100ms_with_timesteps.hdf5 --split_timesteps --timesteps 20

    # ------------- Export accumulated DVS -------------#
    # ipython ./export.py -- ${IN_FULL_FILE_PREFIX}.hdf5 --binsize 0.100 --export_aps 0 --export_dvs 1 --out_file ${OUT_FULL_FILE_PREFIX}_bin100ms_dvs_accum_frames.hdf5

    # Prepare and resize
    # ------------ Prepare APS -------------#
    # ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_frames_100ms.hdf5 --rewrite 1 --skip_mean_std 1
    # ----------- Prepare timestep split DVS ------- #
    # ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_bin100ms_with_timesteps.hdf5 --rewrite 1 --skip_mean_std 1 --split_timesteps --timesteps 20
    # ----------- Prepare accumulated DVS ----------- #
    # ipython ./prepare_cnn_data.py -- --filename ${OUT_FULL_FILE_PREFIX}_bin100ms_dvs_accum_frames.hdf5 --rewrite 1 --skip_mean_std 1

    # ----------- Prepare Encoder Decoder Dataset ----------- #
    # ipython ./prepare_simul_cnn_data.py -- --filename_aps ${OUT_FULL_FILE_PREFIX}_frames_100ms.hdf5 --filename_dvs_split ${OUT_FULL_FILE_PREFIX}_bin100ms_with_timesteps.hdf5 --filename_dvs_accum ${OUT_FULL_FILE_PREFIX}_bin100ms_dvs_accum_frames.hdf5 --rewrite 1 --skip_mean_std 1 --split_timesteps --timesteps 20
done
###BLOCK-COMMENT

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
    dvs_accum_frames_type_list="$dvs_accum_frames_type_list dvs_accum_80x80"
    echo "### Found the following DVS accumulated frames datasets: ${dvs_accum_frames_h5list} ###"
done
# ------------------ Find all timestep seperated DVS datasets ------ #
for filename in ${OUT_DIR}/*_bin100ms_with_timesteps.hdf5
do
    dvs100ms_h5list="$dvs100ms_h5list $filename"
    dvs100ms_type_list="$dvs100ms_type_list dvs_split_80x80"
    echo "### Found the following constant time datasets: ${dvs100ms_h5list} ###"
done

# ------------------ Train Encoder APS to DVS ----------- #
ipython ./multitrain_test_cnn_pytorch.py -- --encoder_decoder --h5files_aps ${frames_h5list[@]} --h5files_dvs ${dvs100ms_h5list[@]} --dataset_keys_aps ${frames_type_list[@]} --dataset_keys_dvs ${dvs100ms_type_list[@]} --run_id encoder_decoder --snn --BNTT --dvs --seperate_dvs_channels --split_timesteps --timesteps 20 --optimizer "SGD" --num_epochs 30
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

# <<'###BLOCK-COMMENT'
# Train the networks

# ------------------ ANN on APS -------------- #
ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${frames_h5list[@]} --dataset_keys ${frames_type_list[@]} --run_id ann_only_aps --optimizer "Adam" --lr 0.001 --batch_size 64 --num_epochs 60 --result_dir ${RESULT_DIR} 

# ------------------ ANN on accumulated DVS -----------#
ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${dvs_accum_frames_h5list[@]} --dataset_keys ${dvs_accum_frames_type_list[@]} --run_id ann_only_acc_dvs --optimizer "Adam" --lr 0.001 --batch_size 64 --num_epochs 60 --result_dir ${RESULT_DIR} 

# ------------------ ANN on APS + Accumulated DVS ---------- #
ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${frames_h5list[@]} ${dvs_accum_frames_h5list[@]} --dataset_keys ${frames_type_list[@]} ${dvs_accum_frames_type_list[@]} --run_id ann_combined_aps_and_acc_dvs --optimizer "Adam" --lr 0.001 --batch_size 64 --num_epochs 60 --result_dir ${RESULT_DIR} 

# ------------------ ANN on encoded DVS -------#
ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${dvs_encoded_h5list[@]} --dataset_keys ${dvs_encoded_type_list[@]} --run_id ann_only_encoded_dvs --optimizer "Adam" --lr 0.001 --batch_size 64 --num_epochs 60 --result_dir ${RESULT_DIR}

# ------------------ ANN on APS + Encoded DVS -------------#
ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${frames_h5list[@]} ${dvs_encoded_h5list[@]} --dataset_keys ${frames_type_list[@]} ${dvs_encoded_type_list[@]} --run_id ann_combined_aps_and_encoded_dvs --optimizer "Adam" --lr 0.001 --batch_size 64 --num_epochs 60 --result_dir ${RESULT_DIR} 
###BLOCK-COMMENT

# ------------------ SNN on APS -------------- #
# ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${frames_h5list[@]} --dataset_keys ${frames_type_list[@]} --run_id aps_snn --snn --BNTT --timesteps 20 --optimizer "Adam" --result_dir ${RESULT_DIR} 

# ------------------ SNN on timestep split DVS ------------#
# ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${dvs100ms_h5list[@]} --dataset_keys ${dvs100ms_type_list[@]} --run_id dvs_snn_timesteps --dvs --snn --BNTT --split_timesteps --seperate_dvs_channels --timesteps 20 --optimizer "Adam" --result_dir ${RESULT_DIR} 

# ------------------ SNN on rate coded DVS --------------- #
# ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${dvs_accum_frames_h5list[@]} --dataset_keys ${dvs_accum_frames_type_list[@]} --run_id dvs_snn_first_conv_coded --timesteps 20 --snn --BNTT --optimizer "Adam" --result_dir ${RESULT_DIR} 

# ------------------ ANN + Encoder (With backprop on encoder) ------- #
# ipython ./multitrain_test_cnn_pytorch.py -- --h5files ${dvs100ms_h5list[@]} --dataset_keys ${dvs100ms_type_list[@]} --run_id ann_back_prop_encoder_only_dvs --dvs --use_encoder --split_timesteps --seperate_dvs_channels --timesteps 20 --optimizer "Adam" --result_dir ${RESULT_DIR} 

<<'###BLOCK-COMMENT'
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
                         --dataset_keys_combined_aps_dvs_enc_frames ${frames_type_list[@]} ${dvs_encoded_type_list[@]}\
                         --timesteps 20\
                         --model_dir ${RESULT_DIR} 

###BLOCK-COMMENT
