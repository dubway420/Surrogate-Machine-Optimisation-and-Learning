echo
echo
echo
echo "================="
echo

case=$1

date
echo "Beginning testing of: " $case

qsub -m ea -M huw.jones@manchester.ac.uk executeTest.sh $case
