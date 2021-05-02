#!/bin/bash

#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=32G                  # memory (per node)
#SBATCH --time=0-8:00            # time (DD-HH:MM)
#SBATCH --output=out.bookmarks.lamp2.log       # output file
#SBATCH --error=err.bookmarks.lamp2.log           # error file

results_dir='/home/muberra/scratch/results_May1_bookmarks_lamp2/'
dataroot='/home/muberra/scratch/data/'
dataset=bookmarks

module load python/3.6 cuda cudnn
source ~/myenv/bin/activate

python -u main.py -dataset $dataset -label_mask 'none' -results_dir $results_dir -dataroot $dataroot
