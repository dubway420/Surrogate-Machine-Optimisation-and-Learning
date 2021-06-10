#!/bin/bash --login
#$ -cwd               # Run the job in the current directory
#$ -l nvidia_v100=1
#$ -t 1-128

experiments=4


experiment_number=$(($SGE_TASK_ID%$experiments+1)) 

sleep $(($experiment_number*20))

date
echo "Starting experiment number: $experiment_number."

# Load modules
module load apps/binapps/anaconda3/2019.07               # Python 3.7.3
module load libs/cuda
module load apps/binapps/keras/2.2.4-tensorflow-gpu      # GPU version of tensorflow 1.14.0
module load libs/intel-18.0/hdf5/1.10.5_mpi              # Intel 18.0.3 compiler, OpenMPI 4.0.1

export OMP_NUM_THREADS=$NSLOTS

python execute_experiment.py $experiment_number $1

echo "Experiment $experiment_number complete."
date
  
  

