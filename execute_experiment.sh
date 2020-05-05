#!/bin/bash --login
#$ -cwd               # Run the job in the current directory
#$ -l nvidia_v100=1

experiment_number=$1

# Load modules
module load apps/binapps/anaconda3/2019.07               # Python 3.7.3
module load libs/cuda
module load apps/binapps/keras/2.2.4-tensorflow-gpu      # GPU version of tensorflow 1.14.0

export OMP_NUM_THREADS=$NSLOTS

python execute_experiment.py $experiment_number