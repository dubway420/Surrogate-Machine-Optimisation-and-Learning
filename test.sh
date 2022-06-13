echo
echo
echo
echo "================="
echo

case==$1

date
echo "Beginning testing of: " $case

qsub -l short -terse executeTest.sh $case