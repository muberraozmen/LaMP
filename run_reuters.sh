#!/bin/bash

#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=8G                  # memory (per node)
#SBATCH --time=0-1:00            # time (DD-HH:MM)
#SBATCH --output=output_reuters_none.log       # output file
#SBATCH --error=err_reuters_none.log           # error file

results_dir='/home/muberra/scratch/results_Apr11_reuters_lamp/'
dataroot='/home/muberra/scratch/data/'
dataset=reuters

module load python/3.6 cuda cudnn
source ~/myenv/bin/activate

python -u main.py -dataset $dataset -label_mask 'none' -results_dir $results_dir -dataroot $dataroot
