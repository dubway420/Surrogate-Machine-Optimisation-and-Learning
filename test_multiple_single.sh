
# Load modules
# module load apps/binapps/anaconda3/2019.07               # Python 3.7.3
# module load libs/cuda
# module load apps/binapps/keras/2.2.4-tensorflow-gpu      # GPU version of tensorflow 1.14.0
# module load libs/intel-18.0/hdf5/1.10.5_mpi              # Intel 18.0.3 compiler, OpenMPI 4.0.1

python3 test_start.py $1

for i in $1/*/; do 
    
    python3 test_single.py $i; 

done

mkdir $1/TEST
mv TEST_$i_* $1/TEST/