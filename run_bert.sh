#!/bin/bash
#SBATCH --output=/home/joot9454/fraud_detection/logs/bert_%j.out
#SBATCH --time=23:59:00
#SBATCH --partition=amilan
#SBACTH --account=ucb305_asc2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20  # up to 64

date
ml slurm/alpine

source /curc/sw/anaconda3/latest
conda activate env

echo "=== Preprocessing ==="
python /home/joot9454/fraud_detection/preprocess.py

echo "=== Fine-tuning BERT ==="
python /home/joot9454/fraud_detection/train_bert.py
date
