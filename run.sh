

line=$(sed '10q;d' experiment_input_files/trial_common_parameters.py)
trial=${line:13}

name=$(echo $trial | tr -d "\"")

mkdir $name

cp -r experiment_input_files $name

J=$(qsub -terse run2.sh $name)

IFS='.'
read -ra ADDR <<< "$J"
JOBID="${ADDR[0]}"

J2=$(qsub -l short -terse -hold_jid $JOBID experiment_evaluation.sh $name)

qsub -l short -terse -hold_jid $J2 -m ea -M huw.jones@manchester.ac.uk cleanup.sh $name $J $J2





