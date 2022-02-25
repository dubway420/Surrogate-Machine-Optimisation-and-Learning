#!/bin/bash --login
#$ -cwd               # Run the job in the current directory

name=$1

jobID1=$2
jobID2=$3

mv $name.ind $name

#mv comparing_models_$name.png $name

mv *.sh.*$jobID1* $name
mv *.sh.*$jobID2* $name

#cp -r experiment_input_files/ $name
