#!/bin/bash

#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=8G                  # memory (per node)
#SBATCH --time=0-0:30            # time (DD-HH:MM)
#SBATCH --output=output_bibtex_prior.log       # output file
#SBATCH --error=err_bibtex_prior.log           # error file

results_dir='/home/muberra/scratch/results_Apr11_bibtex_lamp/'
dataroot='/home/muberra/scratch/data/'
dataset=bibtex

module load python/3.6 cuda cudnn
source ~/myenv/bin/activate

python -u main.py -dataset $dataset -label_mask 'prior' -results_dir $results_dir -dataroot $dataroot
