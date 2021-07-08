#!/bin/bash
#SBATCH --partition=pi_panda
#SBATCH --gpus=rtx2080ti:1
#SBATCH --job-name=agent
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=40G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=josh.chough@yale.edu
#SBATCH --mail-type=ALL

# module load miniconda
# conda activate pytorch_env

# python export.py $1 --time_of_day $2 --binsize 0.025 --separate_dvs_channels --clip
python preprocess.py $1 --time_of_day $2 0.025 --out_dir '/project/panda/shared/ddd20_processed' --disable_tqdm