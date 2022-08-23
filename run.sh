cd ~/parmec_agr_ml_surrogate

email="huw.jones@manchester.ac.uk"

py_files=$(ls experiment_input_files/*.py | wc -l)

# py_files minus one
py_files=$((py_files - 1))

line=$(grep -m 1 "trial_name" experiment_input_files/trial_common_parameters.py)
trial=${line:13}

name=$(echo $trial | tr -d "\"")

for _ in {0..5}; do echo  ; done


echo 
echo >> journal.txt


line="================="

echo $line
echo $line >> journal.txt


date
date >> journal.txt

echo "Beginning trial: " $trial | tr -d '"'
echo "Beginning trial: " $trial | tr -d '"' >> journal.txt

mkdir $name

echo "Folder created"

cp -r experiment_input_files $name

echo "Files copied"

J=$(qsub -terse run2.sh $name $py_files)

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
