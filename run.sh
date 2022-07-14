email="huw.jones@manchester.ac.uk"

line=$(grep -m 1 "trial_name" experiment_input_files/trial_common_parameters.py)
trial=${line:13}

name=$(echo $trial | tr -d "\"")

echo 
echo 
echo 
echo "================="
echo

date
echo "Beginning trial: " $trial | tr -d '"'

mkdir $name

echo "Folder created"

cp -r experiment_input_files $name

echo "Files copied"

J=$(qsub -terse run2.sh $name)

IFS='.'
read -ra ADDR <<< "$J"
JOBID="${ADDR[0]}"

echo "Running trial with job number: " $JOBID  

J2=$(qsub -l short -terse -hold_jid $JOBID experiment_evaluation.sh $name)

qsub -l short -terse -hold_jid $J2 -m ea -M huw.jones@manchester.ac.uk cleanup.sh $name $J $J2

echo "Check on the status of the run at any time with the command: qstat"
echo "Kill the job with the command: qdel" $JOBID 

echo 
echo "When the trial is complete, an email notifying you will be sent to: huw.jones@manchester.ac.uk"
echo
echo "================="
echo
