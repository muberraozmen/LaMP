#!/bin/bash

#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=32G                  # memory (per node)
#SBATCH --time=0-4:00            # time (DD-HH:MM)
#SBATCH --output=output_delicious_prior.log       # output file
#SBATCH --error=err_delicious_prior.log           # error file

results_dir='/home/muberra/scratch/results_Apr7_delicious_prior/'
dataroot='/home/muberra/scratch/data/'
dataset=delicious

module load python/3.6 cuda cudnn
source ~/myenv/bin/activate

python -u main.py -dataset $dataset -label_mask 'prior' -results_dir $results_dir -dataroot $dataroot
