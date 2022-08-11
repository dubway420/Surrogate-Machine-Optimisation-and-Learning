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

makeDir=false
[ -d $name ] && echo "Directory $name exists" || makeDir=true


if $makeDir; then mkdir $name; echo "Folder created"; cp -r experiment_input_files $name; echo "Files copied"; fi

python3 execute_experiment.py $1 $name

python3 experiment_evaluation.py $1