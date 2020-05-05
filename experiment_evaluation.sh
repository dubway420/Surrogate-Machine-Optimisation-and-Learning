#!/bin/bash --login
#$ -cwd               # Run the job in the current directory

# Load modules 
module load apps/binapps/anaconda3/2019.07               # Python 3.7.3

python experiment_evaluation.py third_test
