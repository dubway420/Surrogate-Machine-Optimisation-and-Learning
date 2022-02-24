#!/bin/bash --login
#$ -cwd               # Run the job in the current directory

name=$1

mv $name.ind $name

#mv comparing_models_$name.png $name

mv *.sh.* $name

#cp -r experiment_input_files/ $name
