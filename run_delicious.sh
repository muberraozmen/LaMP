#!/bin/bash

#SBATCH --gres=gpu:2              # Number of GPUs (per node)
#SBATCH --mem=64G                  # memory (per node)
#SBATCH --time=0-4:00            # time (DD-HH:MM)
#SBATCH --output=out.delicious.lamp3.log       # output file
#SBATCH --error=err.delicious.lamp3.log           # error file

results_dir='/home/muberra/scratch/results_May1_delicious_lamp3/'
dataroot='/home/muberra/scratch/data/'
dataset=delicious

module load python/3.6 cuda cudnn
source ~/myenv/bin/activate

python -u main.py -dataset $dataset -label_mask 'prior' -results_dir $results_dir -dataroot $dataroot
